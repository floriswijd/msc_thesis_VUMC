-- public.vitalsign_mat source

CREATE MATERIALIZED VIEW public.vitalsign_mat
TABLESPACE pg_default
AS SELECT ce.subject_id,
    ce.stay_id,
    ce.charttime,
    avg(
        CASE
            WHEN ce.itemid = 220045::numeric AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS heart_rate,
    avg(
        CASE
            WHEN (ce.itemid = ANY (ARRAY[220179::numeric, 220050::numeric, 225309::numeric])) AND ce.valuenum > 0::double precision AND ce.valuenum < 400::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS sbp,
    avg(
        CASE
            WHEN (ce.itemid = ANY (ARRAY[220180::numeric, 220051::numeric, 225310::numeric])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS dbp,
    avg(
        CASE
            WHEN (ce.itemid = ANY (ARRAY[220052::numeric, 220181::numeric, 225312::numeric])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS mbp,
    avg(
        CASE
            WHEN ce.itemid = 220179::numeric AND ce.valuenum > 0::double precision AND ce.valuenum < 400::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS sbp_ni,
    avg(
        CASE
            WHEN ce.itemid = 220180::numeric AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS dbp_ni,
    avg(
        CASE
            WHEN ce.itemid = 220181::numeric AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS mbp_ni,
    avg(
        CASE
            WHEN (ce.itemid = ANY (ARRAY[220210::numeric, 224690::numeric])) AND ce.valuenum > 0::double precision AND ce.valuenum < 70::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS resp_rate,
    round(avg(
        CASE
            WHEN ce.itemid = 223761::numeric AND ce.valuenum > 70::double precision AND ce.valuenum < 120::double precision THEN (ce.valuenum - 32::double precision) / 1.8::double precision
            WHEN ce.itemid = 223762::numeric AND ce.valuenum > 10::double precision AND ce.valuenum < 50::double precision THEN ce.valuenum::double precision
            ELSE NULL::double precision
        END)::numeric, 2) AS temperature,
    max(
        CASE
            WHEN ce.itemid = 224642::numeric THEN ce.value
            ELSE NULL::text
        END) AS temperature_site,
    avg(
        CASE
            WHEN ce.itemid = 220277::numeric AND ce.valuenum > 0::double precision AND ce.valuenum <= 100::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS spo2,
    avg(
        CASE
            WHEN (ce.itemid = ANY (ARRAY[225664::numeric, 220621::numeric, 226537::numeric])) AND ce.valuenum > 0::double precision THEN ce.valuenum
            ELSE NULL::real
        END) AS glucose
   FROM chartevents ce
  WHERE ce.stay_id IS NOT NULL AND (ce.itemid = ANY (ARRAY[220045::numeric, 225309::numeric, 225310::numeric, 225312::numeric, 220050::numeric, 220051::numeric, 220052::numeric, 220179::numeric, 220180::numeric, 220181::numeric, 220210::numeric, 224690::numeric, 220277::numeric, 225664::numeric, 220621::numeric, 226537::numeric, 223762::numeric, 223761::numeric, 224642::numeric]))
  GROUP BY ce.subject_id, ce.stay_id, ce.charttime
WITH DATA;