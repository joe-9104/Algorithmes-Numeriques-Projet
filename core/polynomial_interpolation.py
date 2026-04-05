import numpy as np

# ============================================================
# OPERATIONS SUR LES POLYNOMES
# ============================================================

def poly_add(P, Q):
    n = max(len(P), len(Q))
    R = [0] * n
    for i in range(n):
        if i < len(P):
            R[i] += P[i]
        if i < len(Q):
            R[i] += Q[i]
    return R

def poly_mul(P, Q):
    R = [0] * (len(P) + len(Q) - 1)
    for i in range(len(P)):
        for j in range(len(Q)):
            R[i + j] += P[i] * Q[j]
    return R

def poly_scalar_mul(a, P):
    return [a * coeff for coeff in P]

def poly_eval(P, x):
    s = 0
    for i in range(len(P)):
        s += P[i] * x ** i
    return s


class PolynomialInterpolation:
    def __init__(self, x_points, y_points):
        """
        Initialise l'interpolateur avec les points (x, y).

        Args:
            x_points (list|np.array) : Les abscisses des points.
            y_points (list|np.array) : Les ordonnées des points.

        Raises:
            ValueError: Si les listes ont des tailles différentes ou si les x ne sont pas distincts.
        """

        # Vérification des Longueurs
        if len(x_points) != len(y_points):
            raise ValueError("Les tableaux x_points et y_points doivent avoir la même taille.")

        # Transformation en tableaux numpy
        self.x_points = np.array(x_points, dtype=float)
        self.y_points = np.array(y_points, dtype=float)

        # Vérification de l'unicité des xi
        if np.unique(self.x_points).size != x_points.size:
            raise ValueError("Les points xi doivent être distincts")

        # Initialisation des coefficients de Newton
        self.newton_coeffs = None

    # ============================================================
    # 1) LAGRANGE
    # ============================================================

    def _base_lagrange(self, X, i):
        """Calculer la base de Lagrange Li(x) pour le point i"""
        Li = [1]
        denom = 1
        for j in range(len(X)):
            if j != i:
                # (x - X[j]) est représenté par [-X[j], 1]
                Li = poly_mul(Li, [-X[j], 1])
                denom *= (X[i] - X[j])
        return poly_scalar_mul(1 / denom, Li)

    def _build_lagrange_poly(self, X, Y):
        """Construction du polynôme d'interpolation selon Lagrange"""
        P = [0]
        for i in range(len(X)):
            # Récupérer Li(x) pour chaque i
            Li = self._base_lagrange(X, i)
            # Effectuer l'opération Li(x)*yi
            term = poly_scalar_mul(Y[i], Li)
            # Effectuer l'addition
            P = poly_add(P, term)
        return P

    def lagrange(self, x_eval):
        """Évalue le polynôme de Lagrange au(x) point(s) x_eval."""

        # On construit le polynôme d'interpolation de Lagrange à partir des points donnés
        P = self._build_lagrange_poly(self.x_points, self.y_points)

        # On utilise np.vectorize pour que poly_eval accepte les listes/tableaux d'un coup
        v_poly_eval = np.vectorize(poly_eval, excluded=['P'])

        return v_poly_eval(P=P, x=x_eval)

    # ============================================================
    # 2) NEWTON
    # ============================================================

    def _differences_divisees(self, x, y):
        """Calcule un coefficient par différences divisées (récursivement)"""

        if len(x) == 1:
            return y[0]
        return (self._differences_divisees(x[1:], y[1:]) - self._differences_divisees(x[:-1], y[:-1])) / (x[-1] - x[0])

    def _construction_coefficients_newton(self):
        """Calcule la liste des coefficients f[x0...xi] pour i=0..n-1"""

        self.newton_coeffs = []
        for i in range(len(self.x_points)):
            # On calcule la différence divisée pour les i+1 premiers points
            c = self._differences_divisees(self.x_points[:i + 1], self.y_points[:i + 1])
            self.newton_coeffs.append(c)
        return self.newton_coeffs

    def _build_newton_poly(self):
        """Construction du polynôme d'interpolation de Newton au(x) point(s) x_eval."""

        # Récupérer les coefficients de Newton
        if self.newton_coeffs is None:
            self._construction_coefficients_newton()

        # Récupérer le nombre de points d'interpolation
        n = len(self.x_points)

        # On commence avec le premier terme : f[x0] (un polynôme de degré 0)
        P_final = [self.newton_coeffs[0]]

        # Terme produit courant : (x - x0)(x - x1)...
        produit_accumule = [1]

        for i in range(1, n):
            # 1. On met à jour le produit : produit = produit * (x - x_{i-1})
            # (x - x_{i-1}) est représenté par [-self.x_points[i-1], 1]
            facteur = [-self.x_points[i - 1], 1]
            produit_accumule = poly_mul(produit_accumule, facteur)

            # Calculer le terme complet : f[x0...xi] * (x-x0)...(x-xi-1)
            terme_i = poly_scalar_mul(self.newton_coeffs[i], produit_accumule)

            # Ajouter ce terme au polynôme cumulé
            P_final = poly_add(P_final, terme_i)

        return P_final

    def newton_eval(self, x_eval):
        """Évalue de polynôme de Newton au(x) point(s) x_eval."""

        # On construit le polynôme de Newton à partir des coefficients
        P = self._build_newton_poly()

        # On utilise np.vectorize pour que poly_eval accepte les listes/tableaux d'un coup
        v_poly_eval = np.vectorize(poly_eval, excluded=['P'])

        return v_poly_eval(P=P, x=x_eval)

    def evaluate(self, x_eval, method='newton'):
        """Interface Unifiée"""
        if method.lower().strip() == 'newton':
            return self.newton_eval(x_eval)
        elif method.lower().strip() == 'lagrange':
            return self.lagrange(x_eval)
        else:
            raise ValueError("Méthode d'interpolation inconnue. Choisissez 'newton' ou 'lagrange'.")