-- public.hfnc_hourly_intervals_mat source

CREATE MATERIALIZED VIEW public.hfnc_hourly_intervals_mat
TABLESPACE pg_default
AS WITH hourly_intervals AS (
         SELECT h.stay_id,
            gs.ts AS "timestamp"
           FROM hfnc_episodes_mat h,
            LATERAL generate_series(date_trunc('hour'::text, h.starttime::timestamp without time zone), date_trunc('hour'::text, h.endtime::timestamp without time zone), '01:00:00'::interval) gs(ts)
        )
 SELECT hourly_intervals.stay_id,
    hourly_intervals."timestamp",
    1 AS is_hfnc_active
   FROM hourly_intervals
WITH DATA;