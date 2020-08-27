from settings import GLOB_C, GLOB_K_ON, GLOB_K_OFF, GLOB_K_P, GLOB_DEG_N, GLOB_DEG_M, GLOB_K_F


class Params:
    def __init__(self, c=GLOB_C, k_on=GLOB_K_ON, k_off=GLOB_K_OFF, k_p=GLOB_K_P, d_n=GLOB_DEG_N, d_m=GLOB_DEG_M,
                 k_f=GLOB_K_F):
        self.c = c
        self.k_on = k_on
        self.k_off = k_off
        self.k_p = k_p
        self.x = c * k_on / k_off
        self.pss = self.x / (1 + self.x)
        self.r = c * k_on + k_off
        self.d_n = d_n
        self.d_m = d_m
        self.k_f = k_f

        # two_ligand_kpr
        self.c1 = c
        self.c2 = 0.2 * c
        self.k_off_1 = k_off
        self.k_off_2 = k_off * 3.00

    def unpack(self):
        return self.c, self.k_on, self.k_off, self.k_p, self.x, self.pss, self.r, self.d_n, self.d_m, self.k_f


DEFAULT_PARAMS = Params()

#print(DEFAULT_PARAMS.c)