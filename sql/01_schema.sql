-- Flight-level fact table. One row per scheduled domestic flight.
DROP TABLE IF EXISTS flights;

CREATE TABLE flights (
    flight_date          DATE        NOT NULL,
    year                 SMALLINT    NOT NULL,
    month                SMALLINT    NOT NULL,
    day_of_week          SMALLINT    NOT NULL,   -- 1 = Monday
    dep_hour             SMALLINT    NOT NULL,   -- scheduled local departure hour
    airline              VARCHAR(3)  NOT NULL,
    airline_name         TEXT,
    origin               VARCHAR(4)  NOT NULL,
    origin_city          TEXT,
    dest                 VARCHAR(4)  NOT NULL,
    dest_city            TEXT,
    dep_delay            REAL,
    arr_delay            REAL,
    is_delayed           REAL,                  -- 1 = arrived 15+ min late; NULL if cancelled/diverted
    cancelled            SMALLINT    NOT NULL,
    diverted             SMALLINT    NOT NULL,
    distance             REAL,
    carrier_delay        REAL,
    weather_delay        REAL,
    nas_delay            REAL,
    security_delay       REAL,
    late_aircraft_delay  REAL
);
