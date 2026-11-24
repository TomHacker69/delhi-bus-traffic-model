# Technical Note: Delhi Bus Traffic Model & Journey Planner
**Delhi Transport Stack (Problem 1, Phase 1)**

---

## 1. Linking GPS Points to Routes and Roads

In Delhi, bus GPS pings typically arrive every 5–15 seconds with coordinate jitter, drift, and occasional multi-kilometer multipath glitches. Our map-matching pipeline links raw GPS points to physical routes through a multi-stage spatial filter:

1. **Spatial Bounding & Deduplication**:
   - Out-of-region coordinates (e.g. latitudes outside $28.0^\circ\text{–}29.2^\circ\text{ N}$ or longitudes outside $76.5^\circ\text{–}77.8^\circ\text{ E}$) are immediately dropped using vectorized DuckDB scans.
   - Pings with identical vehicle IDs and timestamps, or non-monotonic timestamps ($t_i \le t_{i-1}$) are filtered out.
2. **Implied Speed Sanity Check**:
   - Between consecutive pings of a vehicle, we compute metric planar displacement $\Delta d$ in meters.
   - If implied speed $v_{implied} = \frac{\Delta d}{\Delta t} \times 3.6 > 80.0\text{ km/h}$, the ping is flagged as an invalid telematics teleportation jump and discarded.
3. **Metric Projection to EPSG:32643**:
   - All spatial calculations are performed in **UTM Zone 43N (EPSG:32643)**, providing uniform Cartesian Euclidean distances in meters across the Delhi National Capital Region without planar distortion.
4. **Route Shape Proximity Matching**:
   - Route shapes are represented as metric `LineString` geometries.
   - Each ping is projected onto its route polyline using vectorized `shapely.distance` and `shapely.line_locate_point`.
   - Pings farther than $50\text{ meters}$ perpendicular to the route shape are dropped.
   - If a `trip_id` is unassigned or corrupted in the telemetry stream, the system uses an in-memory `STRtree` spatial index to associate the bus's median trajectory to the nearest matching route corridor.

---

## 2. Estimating Travel Time per Stretch (Segment)

A **segment** is defined as the directed roadway stretch between two consecutive scheduled stops $(S_k, S_{k+1})$ on a route.

1. **Monotonic Shape Progress**:
   - For every observed trip, pings ordered by timestamp are mapped to cumulative distance along the shape $s_i$.
   - Forward monotonicity is enforced: minor backward jumps ($\le 30\text{ m}$) caused by stationary GPS jitter at stops or traffic lights are clamped to $s_{max}$; large backward jumps ($> 30\text{ m}$) are discarded.
2. **Stop Crossing Time Interpolation**:
   - Each stop in `stops.txt` is projected onto the route shape to obtain exact distance-along-shape $D_k$.
   - The time $T_k$ at which the bus crossed $D_k$ is linearly interpolated between the flanking GPS pings $(t_i, s_i)$ and $(t_{i+1}, s_{i+1})$:
     $$T_k = t_i + (t_{i+1} - t_i) \cdot \frac{D_k - s_i}{s_{i+1} - s_i}$$
3. **Segment Elapsed Time & Filtering**:
   - Segment length: $L = D_{k+1} - D_k$ (meters).
   - Total elapsed duration: $\Delta T = T_{k+1} - T_k$ (seconds).
   - Segments with fewer than 2 pings, non-positive duration, or implausible running speeds ($> 80\text{ km/h}$ or $< 1\text{ km/h}$) are filtered out and cataloged in diagnostic logs.

---

## 3. Dwell Time Treatment

Total time elapsed on a transit stretch consists of **pure running time** and **dwell time** at the boarding stop:

$$\text{Total Time} = \text{Running Time} + \text{Dwell Time}$$

- **Identification**:
  - Pings falling within a $40\text{ meter}$ radius of the stop where reported or implied speed $< 2.0\text{ km/h}$ ($\approx 0.55\text{ m/s}$) are classified as dwelling pings.
  - Dwell duration is computed from the time delta of stationary pings at the stop platform, capped at $80\%$ of total segment time to avoid absorbing traffic jam delays into stop dwell.
- **Dwell Model**:
  - An independent dwell model groups observations by `(stop_id, day_type, hour_bucket)` to learn median passenger boarding/alighting time per stop.
  - High-traffic interchange terminals (e.g. Kashmere Gate ISBT, Dhaula Kuan, Nehru Place Terminal) exhibit longer median dwells ($45\text{–}90\text{ seconds}$), whereas minor roadside stops average $15\text{–}25\text{ seconds}$.
- **Running Time**:
  $$\text{Running Time} = \max(1.0, \Delta T - \text{Dwell Time})$$

---

## 4. Combining Live and Historical Data

When real-time GTFS-RT vehicle position feeds are available (`GTFS_RT_URL`, `GTFS_RT_KEY`):

1. **Current Speed Ratio**:
   - From live pings observed on a segment in the last 15 minutes, current operational speed $v_{live}$ is calculated.
   - The live ratio relative to historical expectation is:
     $$R_{live} = \frac{v_{live}}{v_{hist(segment, day, hour)}}$$
2. **Time-Decay Blending**:
   - For a journey departing in $\Delta t$ minutes ($0 \le \Delta t \le 60$), the real-time condition decays linearly toward historical conditions over a 60-minute horizon:
     $$w_{decay} = \max\left(0.0, 1.0 - \frac{\Delta t}{60.0}\right)$$
     $$R_{effective} = 1.0 + (R_{live} - 1.0) \cdot w_{decay}$$
     $$\text{Running Time}_{blended} = \frac{\text{Running Time}_{historical}}{R_{effective}}$$
   - Ratio $R_{effective}$ is clamped to $[0.4, 2.5]$ to prevent sensor distortion.
3. **Graceful Fallback**:
   - If no live pings exist for a segment, or if GTFS-RT environment variables are missing, the system runs in `historical-only` mode and explicitly marks this in API responses.

---

## 5. Sparse Data Fallback Hierarchy

To ensure 100% network reachability without fabricating arbitrary numbers, the system uses a strictly ordered 4-level fallback hierarchy with confidence tags:

1. **Level 0 (Exact Match / `HIGH` confidence)**:
   - Observed median running time for `(segment_id, day_type, hour_bucket)` with $\ge 3$ historical observations.
2. **Level 1 (Adjacent Hours / `MEDIUM_HIGH` confidence)**:
   - Average of medians for `hour - 1` and `hour + 1` for the same segment and day type.
3. **Level 2 (All-Day Median / `MEDIUM` confidence)**:
   - All-day median for the same segment across that day type (weekday vs weekend).
4. **Level 3 (Route-Level Speed / `LOW` confidence)**:
   - Observed median speed along that entire route for the given hour bucket multiplied by segment length:
     $$\text{Running Time} = \frac{\text{Segment Length}}{v_{route(hour)}}$$
5. **Level 4 (Scheduled GTFS / City Baseline / `VERY_LOW` confidence)**:
   - Scheduled GTFS timetable difference, or empirical Delhi city-wide time-of-day speed curve ($11.5\text{ km/h}$ at 9 AM peak vs $24\text{ km/h}$ at 11 PM night) applied to the segment length.

Every predicted segment returns its `fallback_level` and `confidence` rating, ensuring transparency in both the API and client interfaces.
