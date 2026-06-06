# IMU Subsystem — Heave and Wave Sensing

## Overview

This document describes the design, hardware selection, installation, calibration, and
software architecture for the IMU subsystem of sail-o-vision. The IMU subsystem provides
heave displacement, significant wave height, and wave period data to the Jetson Orin Nano
for use as sea state context for the vision pipeline and for general vessel monitoring.

---

## Purpose and Scope

The IMU subsystem measures the vertical motion of the vessel (heave) caused by wave
action. From the heave time series, the following parameters are derived:

- **Heave displacement** — instantaneous vertical displacement of the hull in meters
- **Significant wave height (Hs)** — mean height of the highest one-third of waves over a
  20-minute window, in meters
- **Mean zero-crossing period (Tz)** — average time between successive upward zero
  crossings of the heave signal, in seconds
- **Peak spectral period (Tp)** — period of the most energetic waves, derived from the
  power spectral density of the heave record, in seconds

Wave direction cannot be determined from a single hull-mounted IMU and is not an output
of this subsystem. Wave direction is obtained from forecast sources (e.g. NOAA WAVEWATCH
III) cross-referenced against the onboard measurements.

The IMU subsystem does **not** replace the Raymarine AR200, which remains the primary
source of vessel heading, pitch, and roll for the navigation network. The AR200 attitude
data (PGN 127257) is available via SignalK and is used as a low-frequency anchor for the
heave Kalman filter.

---

## Hardware

### IMU Breakout Board

**SparkFun 9DoF IMU Breakout — ISM330DHCX + MMC5983MA (Qwiic)**
SparkFun part number: SEN-19895, approximately $34.95.

The ISM330DHCX (STMicroelectronics) provides:

- 3-axis accelerometer: noise density 60 µg/√Hz, temperature range −40 to +105°C
- 3-axis gyroscope: noise density 0.005 °/s/√Hz
- Output data rate: up to 6.7 kHz; operated at 100 Hz for this application
- Interface: I2C or SPI (see Interface Selection below)
- Industrial/automotive grade, designed for vibration-heavy environments

The MMC5983MA (MEMSIC) provides:

- 3-axis magnetometer: 18-bit resolution, no known silicon bugs
- Used for compass heading in the onboard AHRS filter
- Magnetometer heading is supplementary to the AR200; heave computation does not
  depend on it

The 9DoF variant is preferred over the 6DoF (ISM330DHCX only) because it provides
self-contained AHRS capability independent of the AR200, enabling bench testing and
operation as a fallback if the AR200 is unavailable. On an aluminum-hulled vessel the
magnetometer calibration is more challenging than on a GRP vessel (see Calibration
section), but the GRP roof mounting location substantially mitigates this.

### Interface Selection

**Development and bench testing**: I2C via Qwiic connector. The Jetson Orin Nano
Developer Kit carrier board (SP-11324-001) exposes two I2C buses on the 40-pin expansion
header (J12):

- I2C0: SDA pin 27, SCL pin 28 (1.5 kΩ pull-ups on module)
- I2C1: SDA pin 3, SCL pin 5 (2.2 kΩ pull-ups on module)

All 40-pin header signals are 3.3V, matching the Qwiic standard. No level shifting is
required. The ISM330DHCX default I2C address is 0x6A (SA0 low) or 0x6B (SA0 high).
Confirm the available I2C bus number at runtime using `sudo i2cdetect -l` and
`sudo i2cdetect -y <n>`.

Practical I2C cable length limit is approximately 1–3 meters at 100 kHz standard mode,
determined by the 400 pF maximum bus capacitance specified in the I2C specification
(NXP UM10204). The Qwiic pull-ups on both the Jetson module and the SparkFun breakout
are in parallel; verify that the combined pull-up resistance and cable capacitance meet
the rise time budget before extending beyond 1 meter.

**Boat installation**: SPI via shielded CAT6 cable. SPI is a push-pull driven interface
and is significantly more robust to electrical noise from adjacent DC wiring than I2C.
The ISM330DHCX supports SPI natively; switching from I2C to SPI requires only driver
changes — the sensor register map, output data, and all algorithms above the driver
layer are unaffected.

The algorithm and software pipeline are interface-agnostic. Development on I2C and
production on SPI are fully compatible.

### Cabling — Boat Installation

Use shielded CAT6 (S/FTP preferred — individually shielded pairs plus overall shield)
from the sensor mounting location on the GRP roof to the electronics bay housing the
Jetson.

SPI requires 6 conductors. Recommended pair assignment:

| CAT6 Pair | Conductors | Signal | Notes |
|---|---|---|---|
| Pair 1 | orange / white-orange | VCC / GND | Power pair — keep together |
| Pair 2 | green / white-green | MOSI / MISO | Data lines |
| Pair 3 | blue / white-blue | SCK / CS | Clock and chip select |
| Pair 4 | brown / white-brown | spare / spare | Future expansion or second CS |

**Shield grounding**: ground the shield at the Jetson end only. At the sensor end,
terminate the shield conductor but leave it unconnected (insulate or fold back). This
prevents ground loops caused by differences in ground potential between hull bonding
points, which would otherwise drive current through the shield and create the magnetic
interference the shield is intended to prevent.

RJ45 connectors at both ends provide a robust, detachable connection. Use weatherproof
RJ45 boots. Document the pin-out permanently.

---

## Installation Location

### Requirements

The sensor must be mounted:

- On the GRP salon/cockpit roof, on the vessel centerline
- As close as practicable to the longitudinal center of gravity (CoG)
- At least 1 meter from the mast base
- At least 1 meter from MPPT charge controllers (which must be located below decks)
- Clear of solar panel DC wiring runs, or crossed at 90° where runs must be crossed
- Away from concentrations of DC wiring (navigation lights, instrument cables, VHF coax)
- As low as practicable on the roof structure (minimize lever arm above the roll axis)

### Rationale

**GRP roof**: the aluminum hull causes eddy current effects in nearby magnetometers due
to its electrical conductivity. GRP is magnetically and electrically inert, providing a
substantially cleaner magnetic environment for the MMC5983MA magnetometer.

**Centerline**: eliminates port/starboard asymmetry in the magnetic environment from
the aluminum hulls and DC wiring runs in each hull. Equal distance from both hulls means
magnetic contributions from the hull structure tend to cancel. For a catamaran, the
centerline also eliminates lateral lever-arm acceleration from roll, since roll is about
the centerline.

**Near CoG longitudinally**: minimizes rotational acceleration contamination of the
vertical accelerometer signal. When the vessel pitches, every point offset from the CoG
experiences a centripetal and tangential acceleration component that contributes
spuriously to the heave signal. At the CoG this contribution is zero. In practice the
CoG varies with fuel, water, and crew load, so "near the CoG" is the achievable target.

**Distance from mast**: the mast base concentrates wiring (halyards, instruments, VHF
coax, navigation lights) and is a large conductive fitting. The aluminum mast itself
causes eddy current effects. Apply the same 1 meter minimum clearance used by the
AR200 (AR200 Installation Instructions, document 87372 Rev 3, section 7.2).

**MPPT controllers below decks**: MPPT controllers switch current at 20–100 kHz, producing
both magnetic fields from switching currents and RF interference. Their output wiring
carries higher current than the panel wiring. Locating them below decks and away from
the sensor eliminates this interference source entirely.

### Solar Panel Wiring Discipline

Solar panels on the GRP roof produce substantial DC current (potentially 50–100A total
across a full array). The DC wiring is the primary magnetic interference concern, not
the panels themselves (which are silicon and aluminum — neither magnetically significant).

Specify during the build:

- All solar DC wiring on the roof must be run as twisted pair or tightly bundled
  positive/negative conductors for the entire roof run. Twisted pair at a short pitch
  (50–100mm) provides significant cancellation of the magnetic field from the current.
  This is also good practice for an aluminum vessel to minimize hull return currents and
  corrosion risk.
- MPPT charge controllers must be located below decks, not on the roof.
- The sensor I2C/SPI cable must cross solar wiring runs at 90° rather than running
  parallel. Maintain a few centimeters of physical separation where parallel runs are
  unavoidable.

### Location Selection Process (During Build)

1. Obtain the longitudinal CoG position from the designer or builder. Mark it on the
   centerline of the GRP roof structure.
2. Mark exclusion zones: 1m radius around mast base; locations of any planned roof
   penetrations for solar wiring; any planned roof-mounted equipment.
3. Identify the solar panel wiring runs across the roof. These define corridors to avoid
   or cross perpendicularly.
4. Select the point closest to the CoG on the centerline that satisfies all exclusion
   zones.
5. Plan and install a cable conduit from this location to the electronics bay during
   the build. Pulling cable after completion is significantly more difficult.

### Location Validation (After Launch)

With all electrical systems energized (instruments, navigation lights, VHF on standby,
autopilot powered, solar array producing current):

- Walk a quality handheld marine compass around the intended mounting location. Note any
  deflection from expected heading. A stable reading within a few degrees across the area
  indicates acceptable magnetic environment.
- After sensor installation, log raw magnetometer heading while the vessel is stationary
  pointing in a known direction. Rotate the vessel (or use a haul-out turning area) and
  plot measured vs. expected heading. Deviation below 20–25° across the full circle
  indicates a calibratable environment.

---

## Calibration

### What Requires Calibration

The ISM330DHCX accelerometer and gyroscope do not require explicit user calibration:

- Accelerometer DC bias is removed by the high-pass filter in the heave pipeline
- Gyroscope bias is estimated and corrected continuously by the Madgwick AHRS filter

The MMC5983MA magnetometer requires calibration for **hard-iron** and **soft-iron**
distortion:

- **Hard-iron**: constant offset caused by permanently magnetized material on the vessel.
  Shifts the magnetometer measurement sphere off-center. Fixed relative to the vessel.
- **Soft-iron**: distortion caused by magnetically permeable material (non-magnetized
  ferrous metal, in this case primarily the aluminum hull via eddy current effects).
  Squashes the measurement sphere into an ellipsoid with unequal axis scales.

On an aluminum vessel, hard-iron distortion is less severe than on a steel hull
(aluminum does not permanently magnetize), but soft-iron distortion from eddy currents
is present and dynamic — it varies with rate of turn and electrical load. A static
calibration corrects the average distortion. Residual dynamic error is a known limitation.

### Calibration Procedure

The calibration system operates in two modes: normal operation and calibration mode.
Calibration is triggered explicitly by the operator and requires no user input beyond
initiating the procedure and making the required maneuvers.

**Initiation**: operator selects calibration mode via CLI command or web interface.
System logs the start of calibration and begins collecting magnetometer samples.

**Maneuver**: vessel makes two complete 360° circles (720° total) at 3–5 knots in open
water, away from other vessels, marina structures, and underwater cables. The turn should
be slow and steady to sample the full azimuth range evenly. Two full circles provide
redundancy and better coverage than a single circle.

**Data collection**: magnetometer samples collected at 10 Hz throughout the maneuver,
yielding approximately 1,400–2,400 samples for a 2–4 minute procedure.

**Fitting**: on acceptance, an ellipsoid is fitted to the 3D point cloud of magnetometer
samples. The fit extracts:
- Hard-iron offset vector (center of the ellipsoid)
- Soft-iron correction matrix (transforms the ellipsoid back to a sphere)

The fitting computation takes less than one second on the Jetson Orin Nano.

**Acceptance**: the operator reviews the maximum deviation figure. If below approximately
20–25°, the calibration is accepted. If above this threshold, the magnetic environment
at the chosen location is worse than expected and the installation location should be
reconsidered. (The AR200 installation instructions recommend relocation if maximum
deviation at last calibration reaches 45° or above — a more conservative threshold is
appropriate for a raw magnetometer without Raymarine's proprietary linearization.)

**Persistence**: calibration parameters are saved to a JSON file and loaded at startup.
Recalibration is required after any significant change to the vessel's electrical
installation or structural configuration near the sensor.

### Calibration Validity

Calibration is valid for the magnetic environment at the time it was performed. The
following conditions require recalibration:

- New electrical equipment installed near the sensor
- DC wiring rerouted near the sensor
- Significant structural changes to the vessel near the sensor
- Persistent heading errors observed in normal operation

The Earth's magnetic declination varies with geographic position. For heave measurement
this is irrelevant. For heading output, apply a declination correction based on current
GPS position using the NOAA World Magnetic Model (available as `pyIGRF` or `geomag`
Python libraries).

---

## Heave Computation Algorithm

### Inputs

- Raw accelerometer data (ax, ay, az) at 100 Hz from ISM330DHCX, body frame, in g
- Raw gyroscope data (gx, gy, gz) at 100 Hz from ISM330DHCX, body frame, in °/s
- Vessel attitude (roll, pitch, heading) from AR200 via SignalK at ~10 Hz, used as
  low-frequency Kalman anchor

### Pipeline

**Stage 1 — Attitude estimation (AHRS)**

A Madgwick filter fuses the ISM330DHCX gyroscope and accelerometer at 100 Hz to produce
a high-rate attitude estimate (roll φ, pitch θ, heading ψ). AR200 attitude from SignalK
is fused as a low-frequency correction, preventing gyroscope yaw drift and anchoring
the attitude estimate to the navigation network reference.

**Stage 2 — Body-to-earth frame rotation**

The body-frame acceleration vector is rotated to the earth frame using the current
attitude estimate:

    a_earth = R(φ, θ, ψ) × a_body

where R is the ZYX Euler rotation matrix.

**Stage 3 — Gravity removal**

Subtract the gravitational constant from the earth-frame vertical component:

    az_corrected = az_earth − 9.81 m/s²

The result is the net vertical acceleration due to wave motion only. Attitude error at
this stage is the dominant error source in the pipeline: a 1° attitude error produces
approximately 0.17 m/s² of spurious vertical acceleration, which double-integrates to
approximately 0.14 m of false heave at an 8-second wave period.

**Stage 4 — High-pass filter (first)**

A 2nd-order Butterworth high-pass filter with cutoff at 0.04 Hz (25-second period) is
applied to az_corrected. This removes accelerometer DC bias and attitude estimation
drift while passing all ocean swell (typically 0.05–0.5 Hz). Implemented using
`scipy.signal.butter` with `output='sos'` and `scipy.signal.sosfilt` for numerical
stability.

**Stage 5 — First integration**

Numerical integration of filtered acceleration to produce vertical velocity, using the
trapezoidal rule at the IMU sample rate (dt = 0.01 s at 100 Hz).

**Stage 6 — High-pass filter (second)**

The same 0.04 Hz high-pass filter is applied to the velocity signal. This prevents
velocity drift from any residual DC component surviving Stage 4 — the primary mechanism
by which naive double-integration implementations accumulate unbounded displacement
error over time. This dual high-pass scheme is the standard approach used in commercial
wave buoys.

**Stage 7 — Second integration**

Numerical integration of filtered velocity to produce heave displacement in meters.

### Wave Height Statistics

Wave statistics are computed over a 20-minute window (the WMO standard interval),
downsampled from 100 Hz to 4–10 Hz before statistical analysis to reduce memory
requirements.

**Zero-upcrossing analysis**: individual waves are identified by locating successive
upward zero crossings of the mean-subtracted heave record. For each wave (between
successive crossings): wave height H = max − min of the heave segment; wave period T =
time between crossings.

**Significant wave height**: Hs = mean of the highest one-third of individual wave
heights, sorted in descending order. This matches the definition used by operational
oceanography and corresponds to what an experienced observer would report.

**Mean zero-crossing period**: Tz = mean of individual wave periods from the
zero-upcrossing analysis.

**Peak spectral period**: Tp = 1/f_peak, where f_peak is the frequency of maximum power
spectral density computed via Welch's method (`scipy.signal.welch`) on the heave
displacement record, restricted to the wave frequency band 0.04–0.5 Hz.

### Accuracy and Limitations

| Parameter | Expected accuracy | Primary limitation |
|---|---|---|
| Significant wave height Hs | ±20–30% | Vessel RAO; attitude error |
| Mean period Tz | ±0.5–1.0 s | Filter group delay; window length |
| Peak period Tp | ±0.5–1.0 s | Spectral resolution |
| Wave direction | Not available | Single-point measurement |

The vessel does not fully follow short-period waves (period < ~4 s) — a 48-foot
catamaran bridges across short steep chop rather than riding each wave. Heave amplitude
systematically underestimates actual wave height in short-period seas. This is a
fundamental physical limitation, not a sensor or filter issue, and is not correctable
without a vessel heave response amplitude operator (RAO) from tank testing or sea trials
with a reference instrument.

---

## Software Architecture

The IMU subsystem runs as a Python process on the Jetson Orin Nano, alongside the
sail-o-vision vision pipeline. It produces outputs consumed by the vision pipeline for
sea state context and by the SignalK plugin for network distribution.

### Dependencies

- `smbus2` or `spidev` — I2C or SPI transport layer
- `numpy`, `scipy` — signal processing (filtering, integration, spectral analysis)
- `pyIGRF` or `geomag` — magnetic declination correction (optional, for heading output)
- SparkFun `qwiic_ism330dhcx` library (I2C mode) or equivalent SPI driver

### Outputs

- Heave displacement time series (internal, 100 Hz)
- Significant wave height Hs (SignalK, updated every 20 minutes)
- Mean period Tz (SignalK, updated every 20 minutes)
- Peak period Tp (SignalK, updated every 20 minutes)
- Instantaneous roll and pitch (SignalK, 10 Hz, supplementary to AR200)

### Configuration File

Calibration parameters and filter settings are stored in `imu_config.json`:

```json
{
  "interface": "i2c",
  "i2c_bus": 1,
  "i2c_address": "0x6A",
  "sample_rate_hz": 100,
  "highpass_cutoff_hz": 0.04,
  "highpass_order": 2,
  "statistics_window_minutes": 20,
  "hard_iron_offset": [0.0, 0.0, 0.0],
  "soft_iron_matrix": [[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]],
  "madgwick_beta": 0.1
}
```

---

## Development and Testing

### Bench Testing (I2C)

Connect the SparkFun ISM330DHCX Qwiic breakout to the Jetson 40-pin header using a
Qwiic-to-pigtail cable (~200mm):

| Qwiic wire | Jetson J12 pin |
|---|---|
| Red (3.3V) | Pin 1 or 17 |
| Black (GND) | Pin 6, 9, 14, 20, 25, 30, 34, or 39 |
| Blue (SDA) | Pin 3 (I2C1) or Pin 27 (I2C0) |
| Yellow (SCL) | Pin 5 (I2C1) or Pin 28 (I2C0) |

Total bench BOM:

| Item | Part | Price |
|---|---|---|
| IMU | SparkFun 9DoF ISM330DHCX + MMC5983MA (Qwiic) | $34.95 |
| Cable | Qwiic to pigtail, ~200mm | ~$1.50 |
| **Total** | | **~$36.45** |

### Algorithm Validation Sequence

1. **Synthetic injection**: pipe a synthetic wave acceleration signal (JONSWAP spectrum
   or single sinusoid) through the full pipeline and verify that Hs and Tz outputs match
   the known input parameters. This validates filter design before using real hardware.

2. **Static test**: IMU stationary on bench, verify heave output is near zero and does
   not drift over a 20-minute window. Residual drift indicates filter cutoff is too low
   or accelerometer bias is unusually large.

3. **Motion test**: physically rock or swing the IMU at a known frequency and amplitude.
   Verify that the heave output responds at the correct frequency. A playground swing
   (~2–3 second period) produces a realistic sinusoidal motion at the low end of the
   wave frequency band.

4. **Recorded data replay**: obtain a published IMU time series from a wave buoy or
   research vessel dataset. Pipe it through the pipeline and compare Hs/Tz output against
   the published sea state parameters. This is the primary validation step before
   sea trials.

### Transition to SPI (Boat Installation)

Change `"interface": "spi"` in `imu_config.json` and update the driver initialization
call. All signal processing code, calibration parameters, and output formatting are
unchanged. Verify sensor connectivity with a simple register read before full pipeline
startup.

---

## References

- STMicroelectronics ISM330DHCX datasheet (DS13306)
- MEMSIC MMC5983MA datasheet
- NXP I2C-bus specification UM10204 (bus capacitance limits, rise time requirements)
- Raymarine AR200 Installation Instructions, document 87372 Rev 3 (location requirements
  section 7.2; calibration section 12.2)
- NVIDIA Jetson Orin Nano Developer Kit Carrier Board Specification SP-11324-001 v1.3
  (40-pin expansion header Table 3-3; I2C bus pull-up values)
- Madgwick, S.O.H. (2010). An efficient orientation filter for inertial and inertial/
  magnetic sensor arrays. University of Bristol.
- WMO Guide to Wave Analysis and Forecasting, WMO-No. 702 (significant wave height
  definition; 20-minute observation window)
- NOAA World Magnetic Model: https://www.ngdc.noaa.gov/geomag/WMM/