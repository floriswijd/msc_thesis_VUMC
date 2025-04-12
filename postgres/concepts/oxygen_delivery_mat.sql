CREATE MATERIALIZED VIEW public.oxygen_delivery_mat
TABLESPACE pg_default
AS WITH ce_stg1 AS (
         SELECT ce.subject_id,
            ce.stay_id,
            ce.charttime,
                CASE
                    WHEN ce.itemid = ANY (ARRAY[223834::numeric, 227582::numeric]) THEN 223834::numeric
                    ELSE ce.itemid
                END AS itemid,
            ce.value,
            ce.valuenum,
            ce.valueuom,
            ce.storetime
           FROM chartevents ce
          WHERE ce.value IS NOT NULL AND (ce.itemid = ANY (ARRAY[223834::numeric, 227582::numeric, 227287::numeric]))
        ), ce_stg2 AS (
         SELECT ce.subject_id,
            ce.stay_id,
            ce.charttime,
            ce.itemid,
            ce.value,
            ce.valuenum,
            ce.valueuom,
            row_number() OVER (PARTITION BY ce.subject_id, ce.charttime, ce.itemid ORDER BY ce.storetime DESC) AS rn
           FROM ce_stg1 ce
        ), o2 AS (
         SELECT chartevents.subject_id,
            chartevents.stay_id,
            chartevents.charttime,
            chartevents.itemid,
            chartevents.value AS o2_device,
            row_number() OVER (PARTITION BY chartevents.subject_id, chartevents.charttime, chartevents.itemid ORDER BY chartevents.value) AS rn
           FROM chartevents
          WHERE chartevents.itemid = 226732::numeric
        ), stg AS (
         SELECT COALESCE(ce.subject_id, o2.subject_id) AS subject_id,
            COALESCE(ce.stay_id, o2.stay_id) AS stay_id,
            COALESCE(ce.charttime, o2.charttime) AS charttime,
            COALESCE(ce.itemid, o2.itemid) AS itemid,
            ce.value,
            ce.valuenum,
            o2.o2_device,
            o2.rn
           FROM ce_stg2 ce
             FULL JOIN o2 ON ce.subject_id = o2.subject_id AND ce.charttime::text = o2.charttime::text
          WHERE ce.rn = 1
        )
 SELECT stg.subject_id,
    max(stg.stay_id) AS stay_id,
    stg.charttime,
    max(
        CASE
            WHEN stg.itemid = 223834::numeric THEN stg.valuenum
            ELSE NULL::real
        END) AS o2_flow,
    max(
        CASE
            WHEN stg.itemid = 227287::numeric THEN stg.valuenum
            ELSE NULL::real
        END) AS o2_flow_additional,
    max(
        CASE
            WHEN stg.rn = 1 THEN stg.o2_device
            ELSE NULL::text
        END) AS o2_delivery_device_1,
    max(
        CASE
            WHEN stg.rn = 2 THEN stg.o2_device
            ELSE NULL::text
        END) AS o2_delivery_device_2,
    max(
        CASE
            WHEN stg.rn = 3 THEN stg.o2_device
            ELSE NULL::text
        END) AS o2_delivery_device_3,
    max(
        CASE
            WHEN stg.rn = 4 THEN stg.o2_device
            ELSE NULL::text
        END) AS o2_delivery_device_4
   FROM stg
  GROUP BY stg.subject_id, stg.charttime
WITH DATA;