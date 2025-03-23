# EvoWorldAI

Tento skript implementuje DDQN (Double Deep Q-Network) agenta pro hru EvoWorld.io.
Agent se učí hrát pomocí posilovaného učení (Reinforcement Learning, RL).
Využívá neuronovou síť k odhadu hodnot akčních strategií a kombinuje evaluační a target síť,
což řeší problém přeceňování Q-hodnot, typický pro standardní DQN.
Agent vnímá hru jako obrazové vstupy, analyzuje je a provádí akce na základě predikcí své sítě.
Trénuje se s využitím Replay Bufferu a Target Network Update, aby se zlepšila stabilita učení.
Cílem je dosáhnout optimální strategie přežití a evoluce v herním prostředí.
