-- =====================================================================
-- U.S. Flight Delay Intelligence: analysis queries (PostgreSQL)
-- Each query answers one business question. Paste your key results
-- into the README "Findings" section once you've run them.
-- Definitions (same as BTS): a flight is "delayed" if it arrives 15+ minutes
-- late. Delay rate = delayed / completed flights (excludes cancelled and diverted).
-- =====================================================================


-- Q1. Headline KPIs: how reliable is U.S. air travel overall, by year?
SELECT
    year,
    COUNT(*)                                                    AS scheduled_flights,
    ROUND(100.0 * AVG(cancelled), 2)                            AS cancel_rate_pct,
    ROUND(100.0 * AVG(is_delayed), 2)                           AS delay_rate_pct,   -- AVG ignores NULLs
    ROUND(AVG(arr_delay) FILTER (WHERE is_delayed = 1)::numeric, 1) AS avg_minutes_late_when_late
FROM flights
GROUP BY year
ORDER BY year;


-- Q2. Airline scorecard: who is most on time, and how do they rank each year?
WITH airline_year AS (
    SELECT
        year,
        airline_name,
        COUNT(*)                          AS flights,
        AVG(is_delayed)                   AS delay_rate,
        AVG(cancelled)                    AS cancel_rate
    FROM flights
    GROUP BY year, airline_name
)
SELECT
    year,
    airline_name,
    flights,
    ROUND(100 * (1 - delay_rate)::numeric, 1)                          AS on_time_pct,
    ROUND(100 * cancel_rate::numeric, 2)                               AS cancel_pct,
    RANK() OVER (PARTITION BY year ORDER BY delay_rate)                AS on_time_rank
FROM airline_year
ORDER BY year, on_time_rank;


-- Q3. Did each airline improve or get worse year over year? (LAG)
WITH airline_year AS (
    SELECT year, airline_name, AVG(is_delayed) AS delay_rate
    FROM flights
    GROUP BY year, airline_name
)
SELECT
    airline_name,
    year,
    ROUND(100 * delay_rate::numeric, 1)                                         AS delay_rate_pct,
    ROUND(100 * (delay_rate - LAG(delay_rate) OVER w)::numeric, 1)              AS change_pts
FROM airline_year
WINDOW w AS (PARTITION BY airline_name ORDER BY year)
ORDER BY change_pts NULLS LAST;


-- Q4. What causes delays? Share of delay minutes by cause, per airline.
SELECT
    airline_name,
    ROUND(100 * SUM(carrier_delay)       / NULLIF(SUM(arr_delay), 0), 1) AS carrier_pct,
    ROUND(100 * SUM(weather_delay)       / NULLIF(SUM(arr_delay), 0), 1) AS weather_pct,
    ROUND(100 * SUM(nas_delay)           / NULLIF(SUM(arr_delay), 0), 1) AS air_traffic_system_pct,
    ROUND(100 * SUM(security_delay)      / NULLIF(SUM(arr_delay), 0), 1) AS security_pct,
    ROUND(100 * SUM(late_aircraft_delay) / NULLIF(SUM(arr_delay), 0), 1) AS late_aircraft_pct
FROM flights
WHERE is_delayed = 1
GROUP BY airline_name
ORDER BY late_aircraft_pct DESC;


-- Q5. The "snowball effect": does the delay rate grow through the day?
SELECT
    dep_hour,
    COUNT(*)                                  AS flights,
    ROUND(100.0 * AVG(is_delayed)::numeric, 1) AS delay_rate_pct
FROM flights
GROUP BY dep_hour
ORDER BY dep_hour;


-- Q6. Worst routes (minimum 500 flights so small samples don't dominate).
SELECT
    origin || ' -> ' || dest                   AS route,
    COUNT(*)                                   AS flights,
    ROUND(100.0 * AVG(is_delayed)::numeric, 1) AS delay_rate_pct,
    ROUND(AVG(arr_delay) FILTER (WHERE is_delayed = 1)::numeric, 1) AS avg_minutes_late_when_late
FROM flights
GROUP BY origin, dest
HAVING COUNT(*) >= 500
ORDER BY delay_rate_pct DESC
LIMIT 15;


-- Q7. Busiest 20 airports: delay rate plus median and 90th-percentile arrival delay.
SELECT
    origin,
    MIN(origin_city)                                                     AS city,
    COUNT(*)                                                             AS departures,
    ROUND(100.0 * AVG(is_delayed)::numeric, 1)                           AS delay_rate_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY arr_delay)               AS median_arr_delay,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY arr_delay)               AS p90_arr_delay
FROM flights
WHERE cancelled = 0 AND diverted = 0
GROUP BY origin
ORDER BY departures DESC
LIMIT 20;


-- Q8. St. Louis spotlight: how does each airline at STL compare with its own national average?
WITH national AS (
    SELECT airline_name, AVG(is_delayed) AS national_rate
    FROM flights
    GROUP BY airline_name
),
stl AS (
    SELECT airline_name, COUNT(*) AS stl_departures, AVG(is_delayed) AS stl_rate
    FROM flights
    WHERE origin = 'STL'
    GROUP BY airline_name
)
SELECT
    s.airline_name,
    s.stl_departures,
    ROUND(100 * s.stl_rate::numeric, 1)                      AS stl_delay_pct,
    ROUND(100 * n.national_rate::numeric, 1)                 AS national_delay_pct,
    ROUND(100 * (s.stl_rate - n.national_rate)::numeric, 1)  AS stl_vs_national_pts
FROM stl s
JOIN national n USING (airline_name)
ORDER BY s.stl_departures DESC;


-- Q9. Which day of the week is best to fly?
SELECT
    day_of_week,
    TO_CHAR(DATE '2024-01-01' + (day_of_week - 1), 'Day') AS day_name,   -- 2024-01-01 was a Monday
    COUNT(*)                                              AS flights,
    ROUND(100.0 * AVG(is_delayed)::numeric, 1)            AS delay_rate_pct
FROM flights
GROUP BY day_of_week
ORDER BY delay_rate_pct;


-- Q10. STL rolling 7-day delay rate: spot bad stretches (storms, holidays, IT outages).
WITH daily AS (
    SELECT flight_date, COUNT(*) FILTER (WHERE is_delayed IS NOT NULL) AS completed,
           SUM(is_delayed) AS delayed
    FROM flights
    WHERE origin = 'STL'
    GROUP BY flight_date
)
SELECT
    flight_date,
    ROUND(100.0 * SUM(delayed)   OVER w / NULLIF(SUM(completed) OVER w, 0), 1) AS rolling_7d_delay_pct
FROM daily
WINDOW w AS (ORDER BY flight_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY rolling_7d_delay_pct DESC
LIMIT 10;
