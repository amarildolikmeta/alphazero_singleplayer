import particles
import particles.state_space_models as ssm
import particles.distributions as dists


class ToySSM(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.Normal()  # X_0 ~ N(0, 1)

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}
        return dists.Normal(loc=xp)  # X_t ~ N( X_{t-1}, 1)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t (and X_{t-1})
        return dists.Normal(loc=x, scale=self.sigma)  # Y_t ~ N(X_t, sigma^2)
