# Attention_Text

Implementação da utilizacao de atencao para a classificacao de anuncios. O dataset utilizado foi o do mercado livre, disponibilizado para uma competição de classificacao ocorrida em setembro de 2019. Para baixar o dataset apenas cole e rode o link a seguir:

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o2GqP1WTvRi_TYv-qxnZjGlXqboaUOMN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1o2GqP1WTvRi_TYv-qxnZjGlXqboaUOMN" -O train.csv

O modelo utiliza ambos tensorflow e pytorch, para tentar unir o melhor dos dois mundos. Inicialmente os embeddings são treinados nesse vocabulario especifico, e salvos em um vetor numpy. Posteriormente um modelo de classificacao utilizando atenção é utilizado para calssificar os anuncios em diferentes categorias. O modelo utilizado foi baseado neste artigo https://arxiv.org/abs/1703.03130 .

Por fim, pode se ver a atenção em atuacao através desta imagem, onde as palavras com cor de maior intensidade representacao uma maior atenção dada pelo modelo.

![attention](https://github.com/celsolbm/Attention_Text/blob/master/Screen%20Shot%202019-11-07%20at%2014.23.26.png)
