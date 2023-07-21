NEURON {
	POINT_PROCESS gap
	NONSPECIFIC_CURRENT i
	RANGE g, i
	POINTER vgap
}

PARAMETER {
	v (millivolt)
	vgap (millivolt)
	g = 1e-5 (uS)
}

ASSIGNED {
	i (nanoamp)
}

BREAKPOINT {
	i = g*(v - vgap)
}

