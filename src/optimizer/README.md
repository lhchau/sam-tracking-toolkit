# Customized SAM

SAM:

$$
\nabla L^{\text{SAM}}(w)  = \nabla L(w)|_{w + \rho \frac{\nabla L(w)}{||\nabla L(w)||_2}}
$$

Customized SAM:

$$
\nabla L^{\text{Customized SAM}}(w)  = \gamma * \nabla L(w)|_{w + \rho \frac{\nabla L(w)}{||\nabla L(w)||_2}} + (1 - \gamma) * \nabla L(w)
$$