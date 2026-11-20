# Sentimentos em resenhas de livros com BERTimbau

Projeto da disciplina Customizando LLMs, da especialização em IA Generativa da UFPR. O objetivo é classificar frases de resenhas de livros em português como positivas ou negativas e comparar duas estratégias de adaptação do modelo.

## Experimento

O experimento utiliza o [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased), pré-treinado em português brasileiro, com uma nova cabeça de classificação binária. O BERT é um modelo de linguagem mascarada: sua cabeça de pré-treinamento dá lugar a uma cabeça de classificação para a tarefa, em vez de prever palavras. A nova cabeça começa sem treinamento para sentimentos.

A comparação inclui (1) o treinamento apenas da cabeça, com o modelo-base congelado, e (2) o ajuste completo. Os experimentos usam a mesma semente, dados, épocas, lote e taxa de aprendizado. Isso permite uma comparação inicial, mas uma mesma taxa pode favorecer uma estratégia; uma única semente não mede variação entre execuções.

O [corpus ReLi](https://linguateca.pt/Repositorio/ReLi/) contém frases de resenhas anotadas em português. A preparação mantém apenas frases com polaridade positiva ou negativa indicada no corpus e descarta as demais. As partições foram definidas neste projeto por autor: treino (Amado, Meyer, Orwell, Salinger e Sheldon), validação (Rebouças) e teste (Saramago). Elas não são as partições do ReLi-SA. A separação por autor evita compartilhar autores entre partições, mas implica conjuntos de validação e teste pequenos e sujeitos a diferenças de estilo. O resultado mede desempenho em frases com polaridade explícita, não em resenhas inteiras nem em frases neutras ou mistas.

## Ambiente utilizado

A execução registrada neste projeto ocorreu no Linux via WSL, com as seguintes características:

| Componente | Configuração observada |
| --- | --- |
| Processador | AMD Ryzen 7 8745HS; 6 CPUs lógicas disponíveis no ambiente |
| Memória RAM | 7,8 GiB disponíveis para o ambiente WSL |
| GPU | NVIDIA GeForce RTX 4050, 6 GB de VRAM |
| Driver NVIDIA | 591.74 no Windows; CUDA indicada pelo driver: 13.1 |
| Python | 3.11.11 |
| Framework | PyTorch 2.15.0 com CUDA |

Essa é a configuração **testada**, não um requisito mínimo comprovado. No ajuste completo, o pico de memória alocada pelo PyTorch na GPU foi de aproximadamente 2,1 GiB, com lote 8 e sequências de até 128 tokens. O uso total de memória pode ser maior. Há suporte a CPU, mas o desempenho e o consumo de memória nesse modo não foram medidos. O treinamento também depende de memória para o sistema, dados e bibliotecas.

## Como reproduzir

1. Instale Python 3.11 ou superior, [uv](https://docs.astral.sh/uv/getting-started/installation/) e Git. Para usar GPU, configure o driver NVIDIA no sistema e, no caso do WSL, o acesso à GPU a partir do Linux. É necessária conexão com a internet para instalar as dependências e baixar o modelo e os dados.

2. Clone o repositório e instale as versões registradas no arquivo de dependências:

   ```bash
   git clone https://github.com/leandromario/customizando-llms-livros.git
   cd customizando-llms-livros
   uv sync --locked
   ```

3. Verifique se o PyTorch reconhece a GPU. `True` indica que CUDA está disponível para o ambiente Python:

   ```bash
   uv run python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"
   ```

   O script seleciona CUDA automaticamente quando disponível; caso contrário, executa na CPU. A primeira execução pode demorar mais por causa dos downloads.

4. Execute os dois experimentos com a configuração da comparação apresentada abaixo:

   ```bash
   uv run ajustando-llms --modo ambos --epochs 3 --batch-size 8 --max-length 128 --max-train 1000 --learning-rate 0.00002 --seed 42
   ```

5. Consulte `resultados/experimento.json`. O arquivo registra a configuração, a distribuição das classes, as métricas e a matriz de confusão. Os resultados exatos e os tempos podem variar conforme hardware e versões dos drivers.

Para ver todas as opções, execute `uv run ajustando-llms --help`. Use `--max-train 0` para incluir todos os exemplos de treino ou `--modo cabeca` / `--modo completo` para executar uma estratégia individualmente. Em caso de falta de memória, diminua `--batch-size` ou `--max-length`. Os dados são obtidos da fonte original a cada execução; alterações futuras no corpus podem afetar a reprodução.

## Resultados

A execução registra acurácia, F1 macro e matriz de confusão na validação e no teste, parâmetros treináveis, tempo de treino incluindo validação e pico de memória alocada pela GPU. As linhas da matriz representam rótulos reais e as colunas, previsões, na ordem negativo/positivo. O conjunto de teste deve ser reservado para a avaliação final; evite ajustar configurações com base nele. Resultados obtidos em uma RTX 4050 de 6 GB, com 1.000 frases de treino, três épocas, lote 8, semente 42 e a configuração padrão:

| Estratégia | Parâmetros treináveis | Tempo de treino e validação | Pico de GPU | Acurácia no teste | F1 macro no teste |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apenas cabeça | 1.538 | 9,0 s | 450 MiB | 91,78% | 0,527 |
| Modelo completo | 108.924.674 | 39,1 s | 2.103 MiB | 96,22% | 0,867 |

O teste possui 418 frases positivas e 32 negativas: prever sempre positivo daria 92,89% de acurácia. A cabeça congelada acertou somente 2 das 32 negativas; o modelo completo acertou 26. Por isso, F1 macro e matriz de confusão são essenciais para interpretar essa comparação. Os tempos incluem validação a cada época, mas não o download, a carga inicial nem o teste. Foram feitos testes técnicos antes da execução, inclusive no conjunto de teste; portanto, ele não é uma avaliação totalmente intocada. Os resultados são exploratórios, de uma única semente e um único autor no teste. Veja os dados completos em [docs/resultado-experimento.json](docs/resultado-experimento.json).

## Referências

- Souza, Nogueira e Lotufo. *BERTimbau: pretrained BERT models for Brazilian Portuguese* (2020).
- Freitas, Motta, Milidiú e César. *Sparkling Vampire... lol! Annotating Opinions in a Book Review Corpus* (2014).
- [Corpus ReLi: descrição e dados originais](https://linguateca.pt/Repositorio/ReLi/).
