import unittest
from testing import gen_environments as genEnvs
from testing import gen_eval_envs
from stable_baselines3.common.vec_env import VecEnv

# Testing for each of the functions defined in testing.py

class TestRLFunctions(unittest.TestCase) : 

    # FUNCTION 1 -> Environment Generation
    def test_generate_environments(self): 
        '''Asserts the instance type and number of returned environments'''
        a2c, dqn = genEnvs(env_type=0, seed=0)
        self.assertIsInstance(dqn, VecEnv)
        self.assertIsInstance(a2c, VecEnv)

    def test_generate_evalEnvs(self):
        eval_env = gen_eval_envs()
        self.assertIsInstance(eval_env, VecEnv)

    

if __name__ == "__main__" : 
    unittest.main()



