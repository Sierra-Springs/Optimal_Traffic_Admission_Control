change the stepsize function, i.e., c and γ in order to trade-off convergence speed for noise rejection.
What is important, γ or c in this trade-off? What is the effect of c?


# Task 2
Implement the version of the Robinson-Monro algorithm tailored for the problem using decreasing stepsizes using
the standard stepsize εn = c/n^γ, γ ∈ (1/2,1] where c ≥ 0; change the stepsize function, i.e., c and γ in order to
trade-off convergence speed for noise rejection. What is important, γ or c in this trade-off? What is the effect of c?

Plus le discount γ est élevé plus la convergeance est lente.
Plus c est grand et plus la convergeance est rapide.


# Task 3
Implement the version of the Robinson-Monro algorithm tailored for the problem using constant stepsizes;
experiment with different values of ε > 0.

Plus epsilon est grand, plus il y a de fluactuation / perturbation / oscillation dans l'évolution de la politique


# Task 4
Use the version developed for the decreasing stepsizes in combination with Polyak averages;
show what happens with different window sizes. I leave to you how to choose the window for averaging.

Plus la fenêtre de polyak est grande, plus la convergence est lente