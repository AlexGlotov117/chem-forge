from chemicals import Compound, Mixture

Cyclohexane = Compound(name="Cyclohexane", mw=84.16, t_fus=279.84, h_fus=2628)
Urea = Compound(name="Urea", mw=60.056, t_fus=407.2, h_fus=14600.0)

Cyclohexane_Urea = Mixture(compounds=[Cyclohexane, Urea])