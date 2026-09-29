-- Built after the bulk load (much faster than indexing row by row during COPY).
CREATE INDEX IF NOT EXISTS idx_flights_date    ON flights (flight_date);
CREATE INDEX IF NOT EXISTS idx_flights_origin  ON flights (origin);
CREATE INDEX IF NOT EXISTS idx_flights_airline ON flights (airline);
CREATE INDEX IF NOT EXISTS idx_flights_route   ON flights (origin, dest);
ANALYZE flights;
