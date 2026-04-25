import numpy as np
from numpy.polynomial import Polynomial
from typing import TypeAlias, cast


PointArrayLike: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
EvalInput: TypeAlias = float | int | list[float] | tuple[float, ...] | np.ndarray
EvalResult: TypeAlias = float | np.ndarray


class PolynomialInterpolation:
    def __init__(self, x_points: PointArrayLike, y_points: PointArrayLike) -> None:
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
        if np.unique(self.x_points).size != self.x_points.size:
            raise ValueError("Les points xi doivent être distincts")

        # Initialisation des coefficients de Newton
        self.newton_coeffs: list[float] | None = None
        self._lagrange_poly: Polynomial | None = None
        self._newton_poly: Polynomial | None = None

    # ============================================================
    # 1) LAGRANGE
    # ============================================================

    def _base_lagrange(self, X: np.ndarray, i: int) -> Polynomial:
        """Calculer la base de Lagrange Li(x) pour le point i"""
        Li = Polynomial([1.0])
        denom = 1.0
        for j in range(len(X)):
            if j != i:
                # (x - X[j]) en base puissance
                Li *= Polynomial([-X[j], 1.0])
                denom *= (X[i] - X[j])
        return Li / denom

    def _build_lagrange_poly(self, X: np.ndarray, Y: np.ndarray) -> Polynomial:
        """Construction du polynôme d'interpolation selon Lagrange"""
        P = Polynomial([0.0])
        for i in range(len(X)):
            # Récupérer Li(x) pour chaque i
            Li = self._base_lagrange(X, i)
            # Effectuer l'opération Li(x)*yi
            term = Y[i] * Li
            # Effectuer l'addition
            P += term
        return P

    def lagrange(self, x_eval: EvalInput) -> EvalResult:
        """Évalue le polynôme de Lagrange au(x) point(s) x_eval."""

        if self._lagrange_poly is None:
            self._lagrange_poly = self._build_lagrange_poly(self.x_points, self.y_points)
        return cast(EvalResult, self._lagrange_poly(x_eval))

    # ============================================================
    # 2) NEWTON
    # ============================================================

    def _construction_coefficients_newton(self) -> list[float]:
        """Calcule la liste des coefficients f[x0...xi] pour i=0..n-1."""

        n = len(self.x_points)
        dd = self.y_points.astype(float).copy()
        self.newton_coeffs = [dd[0]]

        # Mise à jour in-place des colonnes de différences divisées
        for ordre in range(1, n):
            dd[:n - ordre] = (
                dd[1:n - ordre + 1] - dd[:n - ordre]
            ) / (
                self.x_points[ordre:] - self.x_points[:n - ordre]
            )
            self.newton_coeffs.append(dd[0])

        assert self.newton_coeffs is not None
        return self.newton_coeffs

    def _build_newton_poly(self) -> Polynomial:
        """Construction du polynôme d'interpolation de Newton au(x) point(s) x_eval."""

        # Récupérer les coefficients de Newton
        if self.newton_coeffs is None:
            self._construction_coefficients_newton()

        # Récupérer le nombre de points d'interpolation
        n = len(self.x_points)

        # On commence avec le premier terme : f[x0] (un polynôme de degré 0)
        P_final = Polynomial([self.newton_coeffs[0]])

        # Terme produit courant : (x - x0)(x - x1)...
        produit_accumule = Polynomial([1.0])

        for i in range(1, n):
            # 1. On met à jour le produit : produit = produit * (x - x_{i-1})
            facteur = Polynomial([-self.x_points[i - 1], 1.0])
            produit_accumule *= facteur

            # Calculer le terme complet : f[x0...xi] * (x-x0)...(x-xi-1)
            terme_i = self.newton_coeffs[i] * produit_accumule

            # Ajouter ce terme au polynôme cumulé
            P_final += terme_i

        return P_final

    def newton_eval(self, x_eval: EvalInput) -> EvalResult:
        """Évalue de polynôme de Newton au(x) point(s) x_eval."""

        if self._newton_poly is None:
            self._newton_poly = self._build_newton_poly()
        return cast(EvalResult, self._newton_poly(x_eval))

    def evaluate(self, x_eval: EvalInput, method: str = 'newton') -> EvalResult:
        """Interface Unifiée"""
        if method.lower().strip() == 'newton':
            return self.newton_eval(x_eval)
        elif method.lower().strip() == 'lagrange':
            return self.lagrange(x_eval)
        else:
            raise ValueError("Méthode d'interpolation inconnue. Choisissez 'newton' ou 'lagrange'.")