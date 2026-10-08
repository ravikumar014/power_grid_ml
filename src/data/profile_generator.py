"""
Time-varying load and DER profile generation.

This module generates physically motivated time-series profiles that
will later drive pandapower time-series simulations.

Current profiles:
    - Electrical load demand
    - Solar PV generation

The generated profiles are normalized first and then scaled to the
actual ratings of loads / DER units in the network.

This module does NOT run power flow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ProfileGenerator:
    """
    Generate reproducible time-series profiles for power-system simulation.

    Parameters
    ----------
    start:
        Start timestamp of the simulation.

    periods:
        Number of simulation timesteps.

    freq:
        Pandas frequency string, e.g. "15min".

    seed:
        Random seed used only for small realistic fluctuations.
    """

    def __init__(
        self,
        start: str = "2026-01-01 00:00:00",
        periods: int = 96,
        freq: str = "15min",
        seed: int = 42,
    ) -> None:

        self.start = pd.Timestamp(start)
        self.periods = periods
        self.freq = freq
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        self.timestamps = pd.date_range(
            start=self.start,
            periods=self.periods,
            freq=self.freq,
        )

                                                                        
               
                                                                        

    def time_index(self) -> pd.DatetimeIndex:
        """Return the simulation timestamps."""

        return self.timestamps

                                                                        
                  
                                                                        

    def generate_load_profile(
        self,
        base_load_mw: float = 1.0,
        variability: float = 0.03,
    ) -> pd.DataFrame:
        """
        Generate a normalized daily load profile.

        The profile contains:
            - overnight demand
            - morning increase
            - daytime demand
            - evening peak
            - small stochastic variation

        Parameters
        ----------
        base_load_mw:
            Reference load level.

        variability:
            Magnitude of small stochastic fluctuations.

        Returns
        -------
        pandas.DataFrame
            Columns:
                timestamp
                load_mw
                load_pu
        """

        hours = (
            self.timestamps.hour
            + self.timestamps.minute / 60.0
        )

                                                                        
                     
                                                                        

        profile = np.full(
            shape=len(self.timestamps),
            fill_value=0.72,
            dtype=float,
        )

                                                                        
                                 
                                                                        

        morning_peak = 0.20 * np.exp(
            -0.5 * ((hours - 8.0) / 2.0) ** 2
        )

                                                                        
                      
                                                                        

        evening_peak = 0.35 * np.exp(
            -0.5 * ((hours - 19.0) / 3.0) ** 2
        )

                                                                        
                          
                                                                        

        daytime = 0.08 * np.exp(
            -0.5 * ((hours - 13.0) / 4.0) ** 2
        )

        profile += (
            morning_peak
            + evening_peak
            + daytime
        )

                                                                        
                                    
                                                                        

        noise = self.rng.normal(
            loc=0.0,
            scale=variability,
            size=len(self.timestamps),
        )

        profile += noise

                                              
        profile = np.clip(
            profile,
            a_min=0.1,
            a_max=None,
        )

                                        
        profile_pu = profile / profile.max()

        load_mw = base_load_mw * profile_pu

        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "load_mw": load_mw,
                "load_pu": profile_pu,
            }
        )

                                                                        
                   
                                                                        

    def generate_solar_profile(
        self,
        rated_power_mw: float = 1.0,
        cloud_variability: float = 0.05,
    ) -> pd.DataFrame:
        """
        Generate a normalized solar PV generation profile.

        The profile follows a daylight-shaped curve with small
        fluctuations representing cloud variability.

        Parameters
        ----------
        rated_power_mw:
            Rated PV generation capacity.

        cloud_variability:
            Magnitude of stochastic generation fluctuations.

        Returns
        -------
        pandas.DataFrame
            Columns:
                timestamp
                solar_mw
                solar_pu
        """

        hours = (
            self.timestamps.hour
            + self.timestamps.minute / 60.0
        )

                                                                        
                         
                                                                        

        sunrise = 6.0
        sunset = 18.0

        solar = np.zeros(
            len(self.timestamps),
            dtype=float,
        )

        daylight = (
            (hours >= sunrise)
            & (hours <= sunset)
        )

                                                                        
                            
         
                               
                               
                                                                        

        daylight_fraction = (
            (hours[daylight] - sunrise)
            / (sunset - sunrise)
        )

        solar[daylight] = np.sin(
            np.pi * daylight_fraction
        )

                                                      
        solar[daylight] = solar[daylight] ** 1.5

                                                                        
                                        
                                                                        

        cloud_noise = self.rng.normal(
            loc=0.0,
            scale=cloud_variability,
            size=len(self.timestamps),
        )

        solar += cloud_noise * daylight

        solar = np.clip(
            solar,
            a_min=0.0,
            a_max=1.0,
        )

        solar_mw = rated_power_mw * solar

        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "solar_mw": solar_mw,
                "solar_pu": solar,
            }
        )

                                                                        
                      
                                                                        

    def generate_profiles(
        self,
        base_load_mw: float = 1.0,
        solar_capacity_mw: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate load and solar profiles in a single DataFrame.
        """

        load = self.generate_load_profile(
            base_load_mw=base_load_mw
        )

        solar = self.generate_solar_profile(
            rated_power_mw=solar_capacity_mw
        )

        profiles = load.merge(
            solar,
            on="timestamp",
            how="inner",
        )

        return profiles

                                                                        
          
                                                                        

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> None:
        """
        Save generated profiles to CSV.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            path,
            index=False,
        )


if __name__ == "__main__":

    generator = ProfileGenerator(
        start="2026-01-01 00:00:00",
        periods=96,
        freq="15min",
        seed=42,
    )

    profiles = generator.generate_profiles(
        base_load_mw=1.0,
        solar_capacity_mw=0.5,
    )

    output_path = Path(
        "data/simulated/profiles/profiles.csv"
    )

    generator.save(
        profiles,
        output_path,
    )

    print("\nProfiles generated successfully.\n")

    print(profiles.head(10))

    print("\nProfile statistics:")
    print("-" * 50)

    print(profiles.describe())

    print(
        f"\nSaved to: {output_path}"
    )