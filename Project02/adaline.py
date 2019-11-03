import matplotlib.pyplot as plt


class Adaline(object):
    
    # Construtor
    # LEARNING_RATE  => Taxa de aprendizado (entre 0 e 1)
    # PRECISION      => Precisão requerida
    # RANDOM_WEIGHTS => Pesos sinápticos iniciados com 0 ou aleatóriamente entre 0 e 1
    def _init_(
        self,
        learning_rate = 0.01,
        precision = 0.00001,
        random_weights = False,
    ):
        self.LEARNING_RATE = learning_rate
        self.PRECISION = precision
        self.RANDOM_WEIGHTS = random_weights

    # Método privado que gera o array de pesos sinápticos
    # Podem ser iniciados com 0 ou aleatóriamente entre 0 e 1
    # A posição 0 representará o bias
    def __initialWeights(self, x):
        number_weights = 1 + x.shape[1]
        if (self.RANDOM_WEIGHTS):
            return np.random.random_sample((number_weights,))
        return np.zeros(1 + x.shape[1])
    
    # Treinamento da Rede
    # Sinais de entrada: x e y
    # x => São as propriedades de cada registro
    # y => Determina a qual classe pertence
    def train(self, x, y):
        self.initialWeights = self.__initialWeights(x)
        self.w = list(self.initialWeights)
        
        self.epoca = 0

        eqm_anterior = 0
        eqm_atual = float("inf")

        print('iniciando loop...')
        # print(abs(eqm_atual - eqm_anterior) > self.PRECISION)
        # print(abs(eqm_atual - eqm_anterior))
        # print(eqm_atual - eqm_anterior)
        # print(self.PRECISION)
        while(abs(eqm_atual - eqm_anterior) > self.PRECISION):
            # print('loop...')
            eqm_anterior = self.__eqm(x, y)

            for xi, yi in zip(x, y):
                u = self.__activationPotential(xi)
                # u = self.LEARNING_RATE * (yi - self.__predict(xi))
                
                # self.w[1:] += u * xi
                # self.w[0] += u

                self.w[1:] += self.LEARNING_RATE * (yi - u) * xi
                # self.w[0] += self.LEARNING_RATE * u
                self.w[0] += self.LEARNING_RATE * (yi - u)

                # u = self.LEARNING_RATE * (yi - self.__predict(xi))
                # self.w[1:] += u * xi
                # self.w[0] += u

            self.epoca = self.epoca + 1
            eqm_atual = self.__eqm(x, y)
        print('terminando loop...')
        return self

    # Algoritmo EQM
    def __eqm(self, x, y):
        p = len(x)
        eqm = 0

        for xi, yi in zip(x, y):
            u = self.__activationPotential(xi)
            eqm = eqm + ((yi - u))**2
        eqm = eqm / p
        # print('eqm: %f' % eqm)
        return eqm


    # Validação da Rede
    # Sinais de entrada: x
    # x => São as propriedades de cada registro
    def validation(self, x):
        self.result_x = x
        self.result_y = np.zeros(0)

        # Passa por cada registro
        # xi => Propriedades do registro
        for xi in self.result_x:
            actualResult = self.__predict(xi)
            self.result_y = np.append(self.result_y, actualResult)
        return self

    # Método que calcula o potencial de ativação
    # Retorna a chance de acordo com os pesos de ser de ser uma determinada classe
    def __activationPotential(self, x):
        # self.weights[1:] => Pega o array, exceto o bias
        # Produto do array de propriedades com o array de pesos
        product = np.dot(x, self.w[1:])
        # Produto dos array mais o bias
        # Bias serve para aumentar o grau de liberdade dos ajustes dos pesos
        return product + self.w[0]
    
    # Fase de Operação
    # Método que cálcula a função de ativação
    # Retorna 1 se o novo peso for >= 0 ou -1 se o novo peso for < 0
    def __predict(self, x):
        return np.where(self.__activationPotential(x) >= 0.0, 1, -1)