# Disciplina: Aprendizagem de Máquina
# Laboratório 1 - Impactos da Representação

O script digits.py extrai a represenação mais simples possível de uma base de dados dígitos manuscritos. Para cada posição da imagem, verifica-se o valor de intensidade do pixel e se esse valor for > 128, a característica é igual a 1, caso contrário 0. A imagens tem tamanho variável e como os classificadores precisam de um vetor de tamanho fixo, as imagens são normalizadas utilizando as variáveis X e Y dentro da função rawpixel. Após a execução do programa, um arquivo chamado features.txt é criado no diretório corrente. Esse arquivo contem 2000 linhas no formato

0 0:0 1:0 2:1 3:1

O primeiro caractere indica o rótulo da classe. A sequencia i:v indica o índice da característica e o valor da mesma. Nesse caso, a características 0, 1, 2, e 3 tem valores 0, 0, 1 e 1, respectivamente.

# Autor

* **Darci Luiz Tomasi Junior** - *Initial work* - [darcihp](https://github.com/darcihp)

# Licença

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

