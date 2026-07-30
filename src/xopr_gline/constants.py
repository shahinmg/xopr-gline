"""
Shared physical constants for grounding line detection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Densities (kg m^-3) and the relative dielectric permittivity of ice.
    """

    rho_ice: float = 917.0
    rho_sw: float = 1028.0
    permittivity_ice: float = 3.15

    @property
    def flotation_factor(self) -> float:
        """1 - rho_ice/rho_sw. A floating column of thickness H has surface
        elevation H * flotation_factor."""
        return 1.0 - self.rho_ice / self.rho_sw

    @property
    def density_ratio(self) -> float:
        """rho_sw/rho_ice, used by the height above buoyancy form."""
        return self.rho_sw / self.rho_ice


DEFAULT_CONSTANTS = PhysicalConstants()
