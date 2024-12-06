import time  # Para atrasos
import keyboard  # Para capturar teclas pressionadas
from prettytable import PrettyTable  # Biblioteca para criar tabelas bonitas no terminal
import tracemalloc
from collections import deque
import math
import heapq

labirinto = [
    ['🚩', 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, '🏁', 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
] 

# Posição inicial do ponto
posicao = [0, 0]  # Começa no 'SS'



def imprimir_labirinto(labirinto, posicao):
    """Imprime o labirinto e o ponto atual no terminal."""
    for i, row in enumerate(labirinto):
        for j, cell in enumerate(row):
            if i == posicao[0] and j == posicao[1]:
                print("⭐", end="")  # Ponto móvel
            elif cell == 1:
                print("██", end="")  # Parede
            elif cell == 0:
                print("  ", end="")  # Caminho
            else:
                print(f"{cell}", end="")  # Pontos especiais como 'SS' ou '🏁'
        print()  # Quebra de linha

def mover_ponto(labirinto, posicao, direcao):
    """Calcula a nova posição do ponto móvel."""
    movimentos = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
    nova_posicao = [
        posicao[0] + movimentos[direcao][0],
        posicao[1] + movimentos[direcao][1],
    ]
    if (
        0 <= nova_posicao[0] < len(labirinto)
        and 0 <= nova_posicao[1] < len(labirinto[0])
        and labirinto[nova_posicao[0]][nova_posicao[1]] != 1
    ):
        return nova_posicao
    return posicao

def jogar(labirinto, posicao):
    print("\033[2J\033[H")  # Limpa a tela
    print("🌟 STARS LABYRINTH 🌟")
    print("\nBem-vindo ao Labirinto!")
    print("Você é a estrela (⭐).")
    print("A bandeira vermelha (🚩) é o ponto de partida.")
    print("Chegue à bandeira de chegada (🏁) para vencer.")
    print("\nControles:")
    print("W: Cima | S: Baixo | A: Esquerda | D: Direita")
    print("\nPressione qualquer tecla para começar...")
    keyboard.read_key()

    # Limpa a tela para o jogo
    print("\033[2J\033[?25l")  # Limpa o terminal e esconde o cursor

    # Loop principal
    while True:
        print("\033[H", end="")
        imprimir_labirinto(labirinto, posicao)

        # Aguarda tecla pressionada
        tecla = keyboard.read_key(suppress=True)  # Captura uma tecla por vez
        if tecla in ["w", "s", "a", "d"]:
            posicao = mover_ponto(labirinto, posicao, tecla)
        elif tecla == "q":
            print("\033[2J\033[H")  # Limpa o terminal antes de sair
            print("Jogo encerrado!")
            break
        

        # Limpa o terminal após cada movimento
        print("\033[H", end="")

        # Verifica se chegou ao destino
        if labirinto[posicao[0]][posicao[1]] == '🏁':
            print("\033[2J\033[H")  # Limpa a tela completamente
            print("\nParabéns, você chegou ao destino! 🏆")
            break
        
        time.sleep(0.2)  # Pequeno atraso para suavizar a execução


# Método de busca em profundidade
import tracemalloc

def busca_profundidade(labirinto, inicio, objetivo):
    """Busca em profundidade no labirinto com métricas reais de memória."""
    tracemalloc.start()  # Inicia o monitoramento de memória

    movimentos = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
    pilha = [inicio]
    visitados = set()
    caminho = []
    uso_memoria = 0

    start_time = time.time()  # Inicia o cronômetro

    while pilha:
        uso_memoria = max(uso_memoria, len(pilha))  # Atualiza o uso máximo de nós na pilha
        atual = pilha.pop()
        visitados.add(tuple(atual))
        caminho.append(atual)

        # Verifica se chegamos ao objetivo
        if labirinto[atual[0]][atual[1]] == objetivo:
            end_time = time.time()
            tempo_execucao = end_time - start_time
            tamanho_caminho = len(caminho)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()  # Para o monitoramento de memória

            return {
                "resultado": True,
                "nos_visitados": len(visitados),
                "tempo_execucao": tempo_execucao,
                "tamanho_caminho": tamanho_caminho,
                "uso_memoria": uso_memoria,
                "uso_memoria_real": peak / (1024 * 1024),  # Converte bytes para MB
            }

        # Explora os movimentos possíveis
        for movimento in movimentos.values():
            nova_posicao = [atual[0] + movimento[0], atual[1] + movimento[1]]
            if (
                0 <= nova_posicao[0] < len(labirinto)
                and 0 <= nova_posicao[1] < len(labirinto[0])
                and labirinto[nova_posicao[0]][nova_posicao[1]] != 1
                and tuple(nova_posicao) not in visitados
            ):
                pilha.append(nova_posicao)

    end_time = time.time()
    tempo_execucao = end_time - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Para o monitoramento de memória

    return {
        "resultado": False,
        "nos_visitados": len(visitados),
        "tempo_execucao": tempo_execucao,
        "tamanho_caminho": 0,
        "uso_memoria": uso_memoria,
        "uso_memoria_real": peak / (1024 * 1024),  # Converte bytes para MB
    }


def busca_largura(labirinto, inicio, objetivo):
    """Busca em Largura com rastreamento de métricas."""
    movimentos = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
    fila = deque([(inicio, [inicio])])  # Cada elemento: (posição_atual, caminho_atual)
    visitados = set()

    # Inicia contagem de tempo e memória
    tracemalloc.start()
    inicio_tempo = time.time()

    while fila:
        atual, caminho = fila.popleft()
        visitados.add(tuple(atual))

        # Verifica se chegamos ao objetivo
        if labirinto[atual[0]][atual[1]] == objetivo:
            fim_tempo = time.time()
            memoria_usada, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                "resultado": True,
                "tamanho_caminho": len(caminho),
                "nos_visitados": len(visitados),
                "tempo_execucao": fim_tempo - inicio_tempo,
                "uso_memoria": len(visitados),
                "uso_memoria_real": memoria_usada / (1024 * 1024),
            }

        # Explora movimentos possíveis
        for movimento in movimentos.values():
            nova_posicao = [atual[0] + movimento[0], atual[1] + movimento[1]]
            if (
                0 <= nova_posicao[0] < len(labirinto)
                and 0 <= nova_posicao[1] < len(labirinto[0])
                and labirinto[nova_posicao[0]][nova_posicao[1]] != 1
                and tuple(nova_posicao) not in visitados
            ):
                fila.append((nova_posicao, caminho + [nova_posicao]))
                visitados.add(tuple(nova_posicao))

    # Caso nenhum caminho seja encontrado
    fim_tempo = time.time()
    memoria_usada, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "resultado": False,
        "tamanho_caminho": 0,
        "nos_visitados": len(visitados),
        "tempo_execucao": fim_tempo - inicio_tempo,
        "uso_memoria": len(visitados),
        "uso_memoria_real": memoria_usada / (1024 * 1024),
    }


def heuristica_manhattan(atual, objetivo):
    """Heurística de Manhattan."""
    return abs(atual[0] - objetivo[0]) + abs(atual[1] - objetivo[1])

def heuristica_euclidiana(atual, objetivo):
    """Heurística Euclidiana."""
    return math.sqrt((atual[0] - objetivo[0])**2 + (atual[1] - objetivo[1])**2)

def busca_a_star(labirinto, inicio, objetivo, heuristica):
    """Busca A* com rastreamento de métricas e escolha de heurística."""
    movimentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    inicio = tuple(inicio)

    # Encontra a posição do objetivo
    objetivo_pos = None
    for i in range(len(labirinto)):
        for j in range(len(labirinto[0])):
            if labirinto[i][j] == objetivo:
                objetivo_pos = (i, j)
                break
        if objetivo_pos:
            break

    if not objetivo_pos:
        return {
            "resultado": False,
            "tamanho_caminho": 0,
            "nos_visitados": 0,
            "tempo_execucao": 0,
            "uso_memoria": 0,
            "uso_memoria_real": 0,
        }

    # Inicia contagem de tempo e memória
    tracemalloc.start()
    inicio_tempo = time.time()

    # Inicializa a fila de prioridade
    fila_prioridade = []
    heapq.heappush(fila_prioridade, (0 + heuristica(inicio, objetivo_pos), 0, inicio, [inicio]))
    visitados = set()
    uso_memoria = 0

    while fila_prioridade:
        uso_memoria = max(uso_memoria, len(fila_prioridade))
        _, custo_atual, atual, caminho = heapq.heappop(fila_prioridade)

        if atual == objetivo_pos:
            fim_tempo = time.time()
            memoria_usada, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                "resultado": True,
                "tamanho_caminho": len(caminho),
                "nos_visitados": len(visitados),
                "tempo_execucao": fim_tempo - inicio_tempo,
                "uso_memoria": uso_memoria,
                "uso_memoria_real": memoria_usada / (1024 * 1024),
            }

        visitados.add(atual)

        for movimento in movimentos:
            nova_posicao = (atual[0] + movimento[0], atual[1] + movimento[1])

            if (
                0 <= nova_posicao[0] < len(labirinto)
                and 0 <= nova_posicao[1] < len(labirinto[0])
                and labirinto[nova_posicao[0]][nova_posicao[1]] != 1
                and nova_posicao not in visitados
            ):
                custo = custo_atual + 1
                prioridade = custo + heuristica(nova_posicao, objetivo_pos)
                heapq.heappush(fila_prioridade, (prioridade, custo, nova_posicao, caminho + [nova_posicao]))

    # Caso não encontre o objetivo
    fim_tempo = time.time()
    memoria_usada, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "resultado": False,
        "tamanho_caminho": 0,
        "nos_visitados": len(visitados),
        "tempo_execucao": fim_tempo - inicio_tempo,
        "uso_memoria": uso_memoria,
        "uso_memoria_real": memoria_usada / (1024 * 1024),
    }




def exibir_metricas(metricas):
    """Exibe as métricas em formato de tabela."""
    from prettytable import PrettyTable

    tabela = PrettyTable()
    tabela.field_names = ["Métrica", "Valor"]
    tabela.add_row(["Resultado", "Sim" if metricas["resultado"] else "Não"])
    tabela.add_row(["Total de Nós Visitados", metricas["nos_visitados"]])
    tabela.add_row(["Tempo de Execução", f"{metricas['tempo_execucao']:.6f} segundos"])
    tabela.add_row(["Tamanho do Caminho", metricas["tamanho_caminho"]])
    tabela.add_row(["Uso de Memória (nós)", metricas["uso_memoria"]])
    tabela.add_row(["Uso de Memória (real)", f"{metricas['uso_memoria_real']:.2f} MB"])
    print(tabela)

def menu_principal(labirinto):
    print("\033[2J\033[H")  # Limpa a tela
    print("🌟 Bem-vindo ao Labirinto 🌟")
    print("O que deseja fazer?")
    print("1 - Jogar")
    print("2 - Análise do Jogo")
    opcao = int(input("Digite a opção desejada: "))

    if opcao == 1:
        posicao = [0, 0]  # Define a posição inicial no labirinto
        jogar(labirinto, posicao)
    elif opcao == 2:
        print("\033[2J\033[H")
        print("Qual análise deseja fazer?")
        print("1 - Busca em Profundidade")
        print("2 - Busca em Largura")
        print("3 - Busca A*")
        print("4 - Executar todas as buscas")
        opcao = int(input("Digite a opção desejada: "))

        inicio = [0, 0]
        objetivo = "🏁"

        if opcao == 1:
            print("\033[2J\033[H")
            print("Executando Busca em Profundidade...")
            metricas = busca_profundidade(labirinto, inicio, objetivo)
            exibir_metricas(metricas)
        elif opcao == 2:
            print("\033[2J\033[H")
            print("Executando Busca em Largura...")
            metricas = busca_largura(labirinto, inicio, objetivo)
            exibir_metricas(metricas)
        elif opcao == 3:
            print("\033[2J\033[H")
            print("Executando Busca A*...")
            print("Escolha a heurística:")
            print("1 - Manhattan")
            print("2 - Euclidiana")
            heuristica_opcao = int(input("Digite a opção desejada: "))

            if heuristica_opcao == 1:
                metricas = busca_a_star(labirinto, inicio, objetivo, heuristica_manhattan)
            elif heuristica_opcao == 2:
                metricas = busca_a_star(labirinto, inicio, objetivo, heuristica_euclidiana)
            else:
                print("Opção inválida.")
                return

            exibir_metricas(metricas)

        elif opcao == 4:
            print("\033[2J\033[H")  # Limpa a tela
            print("Executando todas as buscas...\n")

            # Busca em Profundidade
            print("1. Busca em Profundidade:")
            metricas_profundidade = busca_profundidade(labirinto, inicio, objetivo)
            exibir_metricas(metricas_profundidade)

            # Busca em Largura
            print("\n2. Busca em Largura:")
            metricas_largura = busca_largura(labirinto, inicio, objetivo)
            exibir_metricas(metricas_largura)

           

            # Exibir comparação lado a lado para A*
            print("\n3. 📊 Comparação: Busca A* com diferentes heurísticas\n")
            metricas_a_star_euclidiana = busca_a_star(labirinto, inicio, objetivo, heuristica_euclidiana)
            metricas_a_star_manhattan = busca_a_star(labirinto, inicio, objetivo, heuristica_manhattan)
            tabela_comparacao = PrettyTable()
            tabela_comparacao.field_names = ["Métrica", "Heurística Manhattan", "Heurística Euclidiana"]
            tabela_comparacao.add_row(["Resultado",
                                    "Sim" if metricas_a_star_manhattan["resultado"] else "Não",
                                    "Sim" if metricas_a_star_euclidiana["resultado"] else "Não"])
            tabela_comparacao.add_row(["Total de Nós Visitados",
                                    metricas_a_star_manhattan["nos_visitados"],
                                    metricas_a_star_euclidiana["nos_visitados"]])
            tabela_comparacao.add_row(["Tempo de Execução",
                                    f"{metricas_a_star_manhattan['tempo_execucao']:.6f} segundos",
                                    f"{metricas_a_star_euclidiana['tempo_execucao']:.6f} segundos"])
            tabela_comparacao.add_row(["Tamanho do Caminho",
                                    metricas_a_star_manhattan["tamanho_caminho"],
                                    metricas_a_star_euclidiana["tamanho_caminho"]])
            tabela_comparacao.add_row(["Uso de Memória (nós)",
                                    metricas_a_star_manhattan["uso_memoria"],
                                    metricas_a_star_euclidiana["uso_memoria"]])
            tabela_comparacao.add_row(["Uso de Memória (real)",
                                    f"{metricas_a_star_manhattan['uso_memoria_real']:.2f} MB",
                                    f"{metricas_a_star_euclidiana['uso_memoria_real']:.2f} MB"])

            print(tabela_comparacao)

        else:
            print("Opção inválida. Retornando ao menu principal...")

# Executar o menu principal
menu_principal(labirinto)
