# Tutorial científico: ablação de deduplicação fuzzy no MMLU-PT

> **Estado do estudo.** Este documento interpreta os resultados produzidos pelo
> [notebook de ablação](./fuzzy_deduplication_ablation.ipynb) sobre o snapshot
> `541caeb8e48f33ce4b8c8e2e438801f4db5e08347511ffe160bca9970a64b8c5`.
> As conclusões são **provisórias**: os 466 pares selecionados para auditoria
> humana ainda não foram rotulados. Portanto, “candidato”, “aresta” e
> “duplicata sinalizada” não significam, por si sós, duplicata semântica.

## 1. O problema científico

Depois da deduplicação exata, o corpus contém 39.120 questões de múltipla
escolha. Ainda podem restar versões quase iguais de um item: diferenças de
espaçamento, pontuação, cabeçalhos, pequenas revisões ou alternativas
reordenadas impedem uma igualdade byte a byte, embora o conteúdo seja
essencialmente repetido.

A deduplicação fuzzy procura esses casos por similaridade. O benefício é
reduzir vazamento entre treino e avaliação, evitar super-representação de
itens repetidos e melhorar a diversidade. O risco é remover questões que
compartilham vocabulário ou um enunciado-base, mas exigem respostas distintas.
Esse risco é especialmente importante em MCQA (*multiple-choice question
answering*), pois uma pequena mudança em uma alternativa, negação ou unidade
de medida pode mudar o rótulo correto.

O experimento responde a três perguntas:

1. Qual representação do item deve ser comparada: apenas o enunciado ou o
   enunciado acompanhado das alternativas?
2. Quão seletivo deve ser o LSH (*locality-sensitive hashing*)?
3. As decisões resistem a mudanças no tamanho dos n-gramas e na semente do
   MinHash?

O estudo é ablativo porque mantém quase todo o sistema fixo e altera um fator
por vez. Assim, diferenças observadas podem ser atribuídas com mais segurança
à representação, ao banding, aos n-gramas ou à semente.

## 2. Da igualdade exata à similaridade fuzzy

### 2.1 Deduplicação exata

Na deduplicação exata, dois registros são considerados repetidos quando uma
chave canônica coincide. Por exemplo:

```text
Item A: "Qual é a capital do Brasil?"
Item B: "Qual é a capital do Brasil?"
```

Uma comparação direta resolve esse caso. Porém, ela não reconhece:

```text
Item A: "Qual é a capital do Brasil?"
Item B: "Qual a capital do Brasil ?"
```

Normalizar o texto resolve algumas variações, mas não todas. Se a normalização
for agressiva demais, ela também pode apagar diferenças significativas.

### 2.2 Normalização

Normalização transforma variações superficiais em uma forma comparável. São
operações comuns: normalização Unicode, conversão consistente de caixa,
redução de espaços repetidos e padronização de quebras de linha. Considere:

```text
"  QUAL  é a capital?\n"
"qual é a capital?"
```

Após normalização de caixa e espaços, ambas podem se tornar
`"qual é a capital?"`. A normalização deve ser determinística e aplicada antes
da extração dos n-gramas. Ela não deve remover sinais potencialmente
semânticos — por exemplo, o termo “não”, casas decimais ou unidades.

### 2.3 Shingles de caracteres

Um *shingle* é um fragmento contíguo do texto. Neste estudo, cada documento é
representado por um conjunto de n-gramas de caracteres. Com largura 2:

```text
"casa" -> {"ca", "as", "sa"}
"caso" -> {"ca", "as", "so"}
```

Os conjuntos compartilham `{"ca", "as"}`. N-gramas de caracteres toleram
pequenas alterações melhor que a igualdade exata e independem de um
tokenizador linguístico. A largura controla a escala da comparação:

- n-gramas menores geram mais coincidências locais e tendem a elevar a
  recuperação, mas podem confundir textos apenas relacionados;
- n-gramas maiores exigem trechos longos em comum e tendem a ser mais
  seletivos, mas podem perder duplicatas com muitas edições pequenas.

O valor de referência é `char_ngrams=24`; as ablações usam 20 e 30.

### 2.4 Similaridade de Jaccard

Dados dois conjuntos de shingles, \(A\) e \(B\), a similaridade de Jaccard é

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

No exemplo de “casa” e “caso”, a interseção tem 2 elementos e a união tem 4:

\[
J=\frac{2}{4}=0{,}5.
\]

Jaccard igual a 1 indica conjuntos idênticos; 0 indica ausência de shingles em
comum. No notebook, o Jaccard exato é calculado depois que o LSH gera os pares
candidatos. Ele serve como diagnóstico lexical, não como julgamento semântico.

### 2.5 Por que usar MinHash

Calcular Jaccard para todos os pares de 39.120 documentos exigiria

\[
\binom{39120}{2}=765.167.640
\]

comparações. MinHash produz uma assinatura compacta de cada conjunto. Para uma
função hash que induz uma permutação, a probabilidade de dois conjuntos terem
o mesmo mínimo é igual ao Jaccard:

\[
P[h_{\min}(A)=h_{\min}(B)]=J(A,B).
\]

Repetindo o procedimento com várias funções hash, a fração de posições iguais
nas duas assinaturas estima Jaccard. Aqui cada assinatura tem 260 hashes. Mais
hashes reduzem a variância da estimativa, mas aumentam memória e computação.

### 2.6 Bandas LSH e a curva de colisão

Comparar todas as assinaturas ainda seria quadrático. O LSH divide cada
assinatura em \(b\) bandas de \(r\) hashes. Dois documentos tornam-se
candidatos se coincidirem em **todos os hashes de pelo menos uma banda**.

Se a similaridade dos documentos é \(s\), então:

1. a probabilidade de todos os \(r\) hashes de uma banda coincidirem é
   \(s^r\);
2. a probabilidade de uma banda não coincidir é \(1-s^r\);
3. a probabilidade de nenhuma das \(b\) bandas coincidir é
   \((1-s^r)^b\);
4. logo, a probabilidade de pelo menos uma colisão é

\[
P(\text{candidato}\mid s)=1-(1-s^r)^b.
\]

Exemplo: com \(b=2\), \(r=3\) e \(s=0{,}8\),

\[
P=1-(1-0{,}8^3)^2\approx0{,}762.
\]

O ponto `p50` é a similaridade na qual a probabilidade de colisão vale 50%:

\[
s_{50}=\left(1-2^{-1/b}\right)^{1/r}.
\]

As quatro divisões testadas preservam \(b\times r=260\), mas mudam a curva:

| `num_bands` (b) | `minhashes_per_band` (r) | `p50` |
|---:|---:|---:|
| 10 | 26 | 0,901 |
| 13 | 20 | 0,863 |
| 20 | 13 | 0,771 |
| 26 | 10 | 0,695 |

Mantendo 260 hashes, aumentar o número de bandas e reduzir hashes por banda
torna a busca mais permissiva. A interpretação acima decorre da fórmula e da
regra “coincidir em qualquer banda”, e é confirmada empiricamente pelo aumento
dos candidatos. Ela deve prevalecer sobre uma orientação textual da
documentação que sugira o sentido inverso para esse ajuste.

### 2.7 Arestas, componentes conexos e encadeamento

Cada par candidato é uma aresta em um grafo; documentos são vértices. O
workflow reúne vértices ligados direta ou indiretamente em componentes
conexos. Isso permite capturar famílias de versões, mas introduz transitividade:

```text
A -- B -- C
```

Mesmo que A seja parecido com B e B com C, A pode ser pouco parecido com C.
Esse fenômeno é o **encadeamento** (*chaining*). Portanto, “pertencer ao mesmo
componente” não implica que todo par interno ultrapasse um limiar de Jaccard.
No estudo, um componente de seis itens chegou a Jaccard mínimo 0,386 entre
seus membros, embora sua mediana interna fosse 0,415. Outro componente, com 33
itens, teve mínimo 0,533 e mediana 0,731. Esses casos exigem auditoria.

### 2.8 Falsos positivos, falsos negativos e representante

Um falso positivo ocorre quando o método sinaliza como duplicados itens que
são semanticamente distintos. Cabeçalhos padronizados, moldes de prova e
enunciados compartilhados são causas comuns. Um falso negativo ocorre quando
uma duplicata real não colide em nenhuma banda, por exemplo após muitas
edições locais.

Depois de formar um componente, remover todos os itens seria incorreto: é
preciso preservar um representante. Foram comparadas duas políticas, sem
remover dados:

- **equivalente ao workflow:** preserva o representante resultante da ordem
  interna do processamento;
- **determinística por completude:** preserva o item com a representação mais
  longa e desempata pelo ID estável.

Na configuração de referência, as políticas divergiram em 545 dos 817
componentes. A política por completude ganharia, em média, 71,3 caracteres por
componente (mediana 3; máximo 1.075). A quantidade de documentos removidos é a
mesma, mas o conteúdo preservado muda substancialmente.

## 3. Como o NeMo Curator executa o workflow

O experimento usa `FuzzyDeduplicationWorkflow` do NeMo Curator 1.3.0. A visão
conceitual do fluxo é:

```text
Parquet determinístico
        |
        v
shingles de caracteres -> MinHash (260 valores por documento)
        |
        v
bandas LSH -> buckets -> arestas candidatas
        |
        v
componentes conexos -> IDs de documentos sinalizados
        |
        v
auditoria e política de representante (sem aplicar remoção)
```

O estágio MinHash cria assinaturas; o LSH materializa colisões como arestas; o
estágio de componentes conexos agrupa o grafo; e o resultado
`FuzzyDuplicateIds` lista os IDs que seriam removidos. A remoção é uma operação
separada no NeMo Curator e **não foi executada** neste estudo.

Cada tratamento possui cache e resultados exclusivos em
[`../output/fuzzy-deduplication-ablation/runs/`](../output/fuzzy-deduplication-ablation/runs/).
Isso impede que artefatos de configurações diferentes sejam misturados. O
notebook também registra o gerador de IDs, valida sua correspondência com os
IDs estáveis do snapshot e reutiliza resultados completos.

O processamento foi executado via Ray em uma NVIDIA B200, exposta ao processo
como dispositivo lógico 0 por `CUDA_VISIBLE_DEVICES=0`. O ambiente observado
usou Python 3.12.3 e NeMo Curator 1.3.0. Os parâmetros comuns foram
`use_64_bit_hash=False`, `input_blocksize="256MiB"` e
`bands_per_iteration=min(5, num_bands)`. Ray orquestra os estágios e a GPU
acelera as operações CUDA; isso não altera a definição estatística do método.

## 4. Protocolo experimental

### 4.1 Unidade de análise e controle do snapshot

O ponto de partida são 39.120 registros em `output/05 - deduplicated/`, depois
da deduplicação exata. Logo, toda redução medida aqui é **incremental**. O
notebook cria uma entrada Parquet determinística para cada representação,
constrói IDs estáveis e registra o SHA-256 do snapshot. Essa fixação evita que
mudanças silenciosas na entrada sejam confundidas com efeitos dos parâmetros.

### 4.2 Duas representações

- `question`: somente o enunciado, alinhado ao pipeline existente;
- `question_choices`: enunciado seguido das alternativas rotuladas e na ordem
  original. A resposta correta não entra no texto comparado; ela é preservada
  para detectar conflitos posteriormente.

Exemplo sintético:

```text
Enunciado compartilhado: "Qual afirmação é correta?"

Item A: (A) 2 é par; (B) 2 é ímpar.        resposta: A
Item B: (A) 3 é par; (B) 3 é ímpar.        resposta: B
```

Para `question`, os textos são idênticos. Para `question_choices`, a diferença
fica visível. Esse exemplo mostra por que similaridade de enunciado não é
suficiente para declarar equivalência do item completo.

### 4.3 Variáveis controladas e métricas

A referência usa `question_choices`, 20 bandas, 13 hashes por banda,
`char_ngrams=24` e seed 42. As doze execuções cobrem:

- oito combinações de representação e banding;
- duas ablações de n-gramas, 20 e 30, contra 24;
- duas repetições de seed, 17 e 101, contra 42.

As métricas incluem documentos sinalizados, componentes, maior componente,
arestas, Jaccard exato das arestas, abrangência entre exames, distorção por
exame/nível acadêmico, estabilidade entre seeds, tempo e tamanho em disco. O
percentual `J>=0,8` mede a proporção de arestas auditadas com forte sobreposição
lexical; ele é um indicador, não uma estimativa de precisão humana.

## 5. Resultados consolidados

Todos os valores abaixo vêm dos `summary.json` e das saídas salvas no notebook.
`Dup.` é o número de documentos sinalizados para remoção; `Comp.` é o número de
componentes não triviais; `J med.` é a mediana do Jaccard exato das arestas
auditadas. O disco ocupado ficou entre 0,038 e 0,039 GiB por configuração.

| Exp. | Configuração | Rep. | b x r | n | Seed | p50 | Dup. | Comp. | Maior | J med. | J>=0,8 | Tempo (s) | Leitura do tratamento |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| E1 | `question-b10-r26-n24-s42` | Q | 10 x 26 | 24 | 42 | 0,901 | 1.904 | 666 | 53 | 0,943 | 99,393% | 64,64 | Q, banding mais seletivo |
| E2 | `question-b13-r20-n24-s42` | Q | 13 x 20 | 24 | 42 | 0,863 | 2.362 | 813 | 60 | 0,932 | 97,006% | 66,12 | Q, seletividade alta |
| E3 | `question-b20-r13-n24-s42` | Q | 20 x 13 | 24 | 42 | 0,771 | 3.281 | 1.004 | 80 | 0,902 | 85,350% | 73,96 | Q, limiar intermediário |
| E4 | `question-b26-r10-n24-s42` | Q | 26 x 10 | 24 | 42 | 0,695 | 3.655 | 1.111 | 86 | 0,875 | 75,003% | 89,47 | Q, banding mais permissivo |
| E5 | `question_choices-b10-r26-n24-s42` | QC | 10 x 26 | 24 | 42 | 0,901 | 985 | 487 | 26 | 0,921 | 98,081% | 58,95 | QC, banding mais seletivo |
| E6 | `question_choices-b13-r20-n24-s42` | QC | 13 x 20 | 24 | 42 | 0,863 | 1.354 | 596 | 44 | 0,904 | 93,856% | 67,75 | QC, seletividade alta |
| E7 | `question_choices-b20-r13-n24-s42` | QC | 20 x 13 | 24 | 42 | 0,771 | 2.036 | 817 | 53 | 0,862 | 77,665% | 74,38 | Referência principal |
| E8 | `question_choices-b26-r10-n24-s42` | QC | 26 x 10 | 24 | 42 | 0,695 | 2.619 | 937 | 53 | 0,832 | 62,370% | 89,35 | QC, banding mais permissivo |
| E9 | `question_choices-b20-r13-n20-s42` | QC | 20 x 13 | 20 | 42 | 0,771 | 2.153 | 874 | 53 | 0,865 | 77,127% | 74,77 | N-grama mais curto |
| E10 | `question_choices-b20-r13-n30-s42` | QC | 20 x 13 | 30 | 42 | 0,771 | 1.981 | 781 | 51 | 0,860 | 76,653% | 74,10 | N-grama mais longo |
| E11 | `question_choices-b20-r13-n24-s17` | QC | 20 x 13 | 24 | 17 | 0,771 | 2.021 | 821 | 53 | 0,864 | 78,368% | 74,09 | Repetição com seed 17 |
| E12 | `question_choices-b20-r13-n24-s101` | QC | 20 x 13 | 24 | 101 | 0,771 | 2.069 | 819 | 53 | 0,862 | 77,245% | 75,81 | Repetição com seed 101 |

`Q` significa `question`; `QC`, `question_choices`. Percentuais e tempos foram
arredondados somente para apresentação.

## 6. Leitura dos doze experimentos

### 6.1 Banding sobre `question`

**E1 — 10 x 26, p50 0,901.** É o tratamento mais seletivo para enunciados.
Sinalizou 1.904 documentos em 666 componentes. Quase todas as arestas auditadas
(99,393%) têm Jaccard de pelo menos 0,8, mas seletividade lexical elevada não
elimina conflitos semânticos nem encadeamento.

**E2 — 13 x 20, p50 0,863.** Ao tornar a curva menos estrita, foram sinalizados
2.362 documentos em 813 componentes. O maior componente cresceu de 53 para 60,
enquanto `J>=0,8` permaneceu alto, em 97,006%.

**E3 — 20 x 13, p50 0,771.** O salto para 3.281 documentos e 1.004 componentes
mostra que muitos enunciados estão na faixa intermediária de similaridade. A
proporção `J>=0,8` caiu para 85,350% e o maior componente chegou a 80.

**E4 — 26 x 10, p50 0,695.** É a configuração mais agressiva do estudo: 3.655
documentos, 1.111 componentes e maior componente de 86 itens. Apenas 75,003%
das arestas auditadas alcançam Jaccard 0,8, sinal de maior cobertura acompanhada
de risco mais alto.

### 6.2 Banding sobre `question_choices`

**E5 — 10 x 26, p50 0,901.** Ao exigir semelhança também nas alternativas, o
número de sinalizações cai para 985, em 487 componentes. É o mínimo do sweep e
tem `J>=0,8` de 98,081%.

**E6 — 13 x 20, p50 0,863.** Recupera 1.354 documentos em 596 componentes. O
maior componente cresce para 44 e a fração `J>=0,8` cai para 93,856%, ainda
mantendo maior seletividade que os bandings seguintes.

**E7 — 20 x 13, p50 0,771.** Esta é a referência para as demais ablações:
2.036 documentos, 817 componentes e maior componente de 53. Sua mediana de
Jaccard é 0,862, mas só 77,665% das arestas alcançam 0,8. É um ponto de
compromisso a ser avaliado, não uma escolha já aprovada.

**E8 — 26 x 10, p50 0,695.** O tratamento permissivo encontra 2.619 documentos
em 937 componentes. A mediana de Jaccard cai para 0,832 e `J>=0,8` para
62,370%, a menor proporção observada. A cobertura adicional precisa ser
justificada por rótulos humanos.

### 6.3 Ablação de `char_ngrams`

**E9 — n=20 contra n=24.** Com banding e seed fixos, reduzir o shingle para 20
caracteres aumenta as sinalizações de 2.036 para 2.153 e os componentes de 817
para 874. O resultado é coerente com maior tolerância a alterações locais. O
efeito é moderado em comparação com mudar banding ou representação.

**E10 — n=30 contra n=24.** Aumentar o shingle reduz as sinalizações para 1.981
e os componentes para 781. O maior componente também cai de 53 para 51. A
direção confirma maior seletividade, embora a mediana de Jaccard das arestas
selecionadas (0,860) permaneça muito próxima à referência (0,862).

### 6.4 Ablação de seed

**E11 — seed 17 contra seed 42.** A repetição encontra 2.021 documentos em 821
componentes, próxima aos 2.036 e 817 da referência. Contudo, proximidade nas
contagens agregadas não garante os mesmos documentos.

**E12 — seed 101 contra seed 42.** Esta repetição encontra 2.069 documentos em
819 componentes. O maior componente permanece em 53 nas três seeds, mas a
identidade dos candidatos varia.

O Jaccard dos conjuntos de documentos agrupados foi 0,805 entre seeds 42 e 17,
0,794 entre 42 e 101 e 0,781 entre 17 e 101. Assim, cerca de 20% da união pode
mudar conforme a aleatoriedade do MinHash. Essa estabilidade é razoável para
um filtro aproximado, mas insuficiente para tratar uma única execução como
verdade determinística.

## 7. O que os resultados ensinam

### 7.1 A representação muda mais que um detalhe de implementação

Para cada banding, incluir as alternativas reduz o número de documentos
sinalizados:

| Banding | Somente questão | Questão + alternativas | Redução ao usar QC |
|---:|---:|---:|---:|
| 10 x 26 | 1.904 | 985 | 48,3% |
| 13 x 20 | 2.362 | 1.354 | 42,7% |
| 20 x 13 | 3.281 | 2.036 | 37,9% |
| 26 x 10 | 3.655 | 2.619 | 28,3% |

Isso revela muitos enunciados compartilhados com alternativas diferentes. A
representação `question` é útil para encontrar reutilização de moldes; a
representação `question_choices` está mais próxima da identidade do item MCQA.
Nenhuma das duas incorpora a resposta correta ao texto do MinHash, de modo que
conflitos de resposta continuam sendo um sinal de auditoria independente.

Também houve diferença por origem. Nas quatro configurações `question`, foram
observados 1, 2, 5 e 6 componentes entre exames à medida que o banding se tornou
mais permissivo. Nas configurações `question_choices`, não houve componentes
entre exames. Isso não prova ausência de duplicatas entre exames: indica apenas
que nenhuma foi recuperada sob esses tratamentos.

### 7.2 Permissividade aumenta recuperação e incerteza

O intervalo global foi de 985 a 3.655 documentos sinalizados. Ao reduzir o
`p50`, a quantidade cresce e a proporção de arestas com Jaccard maior ou igual
a 0,8 cai: de 99,393% para 75,003% em `question`, e de 98,081% para 62,370% em
`question_choices`. Isso é o comportamento esperado de um detector que amplia
sua região de recuperação.

Não se deve interpretar `p50=0,771` como um limiar rígido. Pares abaixo desse
valor ainda podem colidir e pares acima podem não colidir. O `p50` descreve uma
probabilidade, não uma regra determinística.

### 7.3 Trechos reais da auditoria

Os exemplos seguintes são deliberadamente curtos, têm IDs abreviados e não
reproduzem questões completas. Todos estão **sem rótulo humano**; servem para
mostrar o tipo de evidência, não para antecipar a classe correta.

- Par `2abd5439…::7e7bdbd…`, mesmo exame
  `RESIDENCIA_USP_UNICAMP`, Jaccard 0,960, componente de tamanho 4. Ambos começam
  com “Mulher, 32 anos de idade, assintomática, é encaminhada para avaliação de
  nódulo hepático…”. As respostas e alternativas registradas coincidem. É um
  candidato de alta similaridade que ainda requer confirmação humana.
- Par `29cbffa8…::c49e2b86…`, exame `OBI`, Jaccard 0,796, componente de tamanho
  4. Ambos começam com “Prova de Matemática Num período de três dias, de
  segunda a quarta-feira…”. As respostas corretas e alternativas diferem. O
  caso ensina por que um enunciado-base comum pode esconder tarefas distintas.
- Par `13650562…::b34559c5…`, exame `BACEN`, Jaccard 0,476, componente de
  tamanho 20. Ambos começam com o cabeçalho “Nas questões a seguir, marque,
  para cada uma, a única opção correta…”. O conteúdo compartilhado pode ser
  apenas instrução de prova, um padrão típico a investigar em configurações
  permissivas e componentes encadeados.

### 7.4 Distorção do corpus

A maior taxa de remoção observada para um exame variou de 12,252% no tratamento
QC mais seletivo a 75,253% no tratamento Q mais permissivo. A maior mudança de
participação de um exame no corpus foi de 0,626 a 3,304 pontos percentuais.
Assim, maximizar o total removido pode alterar desproporcionalmente a composição
do benchmark. O desempate entre configurações deve preferir menor distorção.

### 7.5 Encadeamento e representante são decisões independentes

O Jaccard baixo dentro de alguns componentes mostra que a aresta local e o
agrupamento global respondem a perguntas diferentes. Uma política conservadora
pode exigir uma verificação adicional entre cada membro e o representante, ou
dividir componentes heterogêneos, antes de remover qualquer registro.

Além disso, a divergência de representante em 545 de 817 componentes da
referência mostra que “qual exemplar fica” é uma decisão metodológica, não um
detalhe administrativo. Preservar o item mais completo é reprodutível e tende
a conservar contexto, mas comprimento pode também preservar ruído ou
cabeçalhos. A auditoria deve verificar essa hipótese.

### 7.6 Custo do sistema

As execuções levaram aproximadamente 59 a 89 segundos e ocuparam 0,038 a 0,039
GiB cada. O custo aumenta nos bandings permissivos porque mais colisões geram
mais arestas: as execuções produziram de 2.401 a 39.776 arestas brutas. Nesse
snapshot e na B200, custo não parece ser o critério dominante; precisão e
distorção são mais importantes. Tempos não devem ser extrapolados para outro
hardware ou volume sem nova medição.

## 8. Fronteira de Pareto e auditoria humana

Uma configuração domina outra quando remove pelo menos tanta redundância sem
piorar os critérios de qualidade/custo considerados, e melhora ao menos um
deles. A **fronteira de Pareto** contém os tratamentos não dominados: aumentar
um benefício exige aceitar perda em outro. O notebook usa essa fronteira para
evitar gastar anotação com alternativas claramente inferiores.

Pelos dois objetivos automáticos usados na triagem — mais documentos
sinalizados e maior proporção `J>=0,8` — a fronteira observada contém E5, E6,
E7, E8 e E9. Isso não torna esses tratamentos semanticamente válidos; apenas
justifica submetê-los à próxima etapa de avaliação.

Foram amostrados deterministicamente 466 pares, estratificados por faixa de
similaridade, tamanho do componente e origem intra/inter-exame. O arquivo
[`manual_labels.csv`](../output/fuzzy-deduplication-ablation/manual_labels.csv)
preserva rótulos existentes e aceita quatro classes:

- `duplicate`: os itens avaliam essencialmente a mesma questão e um deles pode
  representar o outro;
- `related_but_distinct`: compartilham contexto, molde ou conteúdo, mas são
  tarefas diferentes;
- `distinct`: não há equivalência relevante;
- `unsure`: a evidência disponível não permite decisão confiável.

Para estimar a precisão humana de uma configuração, `duplicate` conta como
sucesso e as classes semânticas negativas contam como erro; `unsure` deve ser
reportado separadamente, não convertido silenciosamente em acerto. Para \(k\)
duplicatas confirmadas em \(n\) decisões válidas, \(\hat p=k/n\). O intervalo
de Wilson de 95%, com \(z=1{,}96\), é

\[
\frac{\hat p+z^2/(2n)\ \pm\ z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}
     {1+z^2/n}.
\]

Wilson é preferível ao intervalo normal simples quando a amostra é pequena ou
a precisão está perto de 0 ou 1. O critério conservador do estudo exige limite
inferior de Wilson de pelo menos 95% **e todos os pares solicitados para aquela
configuração com rótulo decisivo**. Entre configurações elegíveis, recomenda a
que remove mais redundância; os desempates favorecem menor distorção entre
exames, maior estabilidade entre seeds e menor custo.

No estado atual, as 466 linhas estão sem rótulo. Portanto, não há estimativa de
precisão humana, nenhuma configuração pode ser declarada elegível e **não
existe evidência para uma recomendação final**.

## 9. Guia de interpretação depois da anotação

1. Validar os rótulos e separar `unsure` das decisões válidas.
2. Calcular precisão e intervalo de Wilson por configuração, preservando os
   estratos da amostra.
3. Eliminar configurações cujo limite inferior seja menor que 95%.
4. Entre as elegíveis, maximizar documentos duplicados confirmados, não apenas
   sinalizações automáticas.
5. Inspecionar componentes grandes, conflitos de resposta e pares com Jaccard
   baixo; eles concentram risco de encadeamento.
6. Comparar a distribuição por exame e nível acadêmico antes e depois da
   remoção simulada.
7. Escolher explicitamente a política de representante e registrar a decisão.
8. Reexecutar a configuração candidata em outra seed e revisar as diferenças.
9. Só então integrar uma remoção ao pipeline, em mudança separada e reversível.

Resultados possíveis devem ser lidos assim:

- **Alta precisão e pouca remoção:** configuração segura, mas possivelmente
  conservadora; revisar falsos negativos.
- **Alta remoção e precisão insuficiente:** detector útil para triagem, não para
  remoção automática.
- **Boa média e limite inferior baixo:** faltam exemplos rotulados ou há
  heterogeneidade; ampliar a auditoria.
- **Boa precisão, mas grande distorção por exame:** reconsiderar o banding ou
  aplicar regras por subgrupo.
- **Instabilidade entre seeds:** aumentar a assinatura, usar consenso entre
  execuções ou validar pares por Jaccard exato antes de decidir.

## 10. Ameaças à validade

### Validade de construto

Jaccard de n-gramas mede sobreposição lexical, não equivalência pedagógica. A
resposta correta foi usada como sinal de conflito, mas não integrou a
assinatura. Comprimento é apenas um proxy de completude do representante.

### Validade interna

MinHash é aleatório e a estabilidade observada, 0,781–0,805, não é perfeita.
Componentes conexos adicionam transitividade. A auditoria deriva de candidatos
gerados pelo próprio sistema e, portanto, não estima diretamente falsos
negativos no corpus inteiro.

### Validade externa

Os resultados pertencem a este snapshot, idioma, domínio de exames e formato
MCQA. A B200 afeta custo, não garante o mesmo tempo em outro ambiente. A faixa
de parâmetros é compacta e não cobre todo o espaço possível.

### Validade estatística

Os 466 pares ainda não têm rótulos. Após a anotação, dependência entre pares do
mesmo componente e tamanhos diferentes de estrato devem ser considerados na
interpretação. Contagens de sinalizações não substituem precisão ou recall.

### Validade operacional

Resultados em cache podem ficar incompletos após interrupção; o notebook valida
artefatos antes de reutilizá-los. IDs estáveis e fingerprint reduzem, mas não
eliminam, riscos de desalinhamento de versões e entradas.

## 11. Checklist para uma decisão defensável

- [ ] As 466 amostras foram rotuladas por instruções consistentes?
- [ ] Casos `unsure` e desacordos entre anotadores foram reportados?
- [ ] O limite inferior de Wilson é pelo menos 95%?
- [ ] Há auditoria específica de conflitos de alternativas e respostas?
- [ ] Componentes grandes e com baixo Jaccard mínimo foram examinados?
- [ ] A distribuição por exame e nível acadêmico permanece aceitável?
- [ ] A decisão é estável entre seeds?
- [ ] A política de representante foi escolhida e justificada?
- [ ] O snapshot e as versões do ambiente foram registrados?
- [ ] A remoção será aplicada somente depois de uma simulação revisável?

Até que esses itens sejam satisfeitos, os resultados devem orientar investigação
e anotação, não exclusão automática de dados.

## 12. Reprodutibilidade e referências

Os resultados intermediários ficam em `output/`, fora do versionamento. Para
reproduzir o estudo, use o
[notebook](./fuzzy_deduplication_ablation.ipynb), confira o snapshot e ative a
execução somente no ambiente CUDA pretendido. Cada `summary.json` registra os
parâmetros, tempos, volume em disco e fingerprint da entrada.

- NVIDIA. [NeMo Curator: Fuzzy Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/fuzzy).
- NVIDIA. [NeMo Curator: Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/).
- Broder, A. Z. (1997). [On the resemblance and containment of documents](https://doi.org/10.1109/SEQUEN.1997.666900).
- Leskovec, J.; Rajaraman, A.; Ullman, J. D. [*Mining of Massive Datasets* — capítulo sobre LSH](https://infolab.stanford.edu/~ullman/mmds/book.pdf).
