from nano_aevlv.envs import SudokuEnv

if __name__ == "__main__":
    env = SudokuEnv()
    env.reset(1337, difficulty=0.25)
    print(env.board)
    score = env.step([0, 5, 5])
    print(score)
    print(env.board)
