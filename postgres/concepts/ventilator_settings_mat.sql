-- public.ventilator_settings_mat source

CREATE MATERIALIZED VIEW public.ventilator_settings_mat
TABLESPACE pg_default
AS WITH ce AS (
         SELECT ce_1.subject_id,
            ce_1.stay_id,
            ce_1.charttime,
            ce_1.itemid,
            ce_1.value,
                CASE
                    WHEN ce_1.itemid = 223835::numeric THEN
                    CASE
                        WHEN ce_1.valuenum >= 0.20::double precision AND ce_1.valuenum <= 1::double precision THEN ce_1.valuenum * 100::double precision
                        WHEN ce_1.valuenum > 1::double precision AND ce_1.valuenum < 20::double precision THEN NULL::double precision
                        WHEN ce_1.valuenum >= 20::double precision AND ce_1.valuenum <= 100::double precision THEN ce_1.valuenum::double precision
                        ELSE NULL::double precision
                    END
                    WHEN ce_1.itemid = ANY (ARRAY[220339::numeric, 224700::numeric]) THEN
                    CASE
                        WHEN ce_1.valuenum > 100::double precision THEN NULL::real
                        WHEN ce_1.valuenum < 0::double precision THEN NULL::real
                        ELSE ce_1.valuenum
                    END::double precision
                    ELSE ce_1.valuenum::double precision
                END AS valuenum,
            ce_1.valueuom,
            ce_1.storetime
           FROM chartevents ce_1
          WHERE ce_1.value IS NOT NULL AND ce_1.stay_id IS NOT NULL AND (ce_1.itemid = ANY (ARRAY[224688::numeric, 224689::numeric, 224690::numeric, 224687::numeric, 224685::numeric, 224684::numeric, 224686::numeric, 224696::numeric, 220339::numeric, 224700::numeric, 223835::numeric, 223849::numeric, 229314::numeric, 223848::numeric, 224691::numeric]))
        )
 SELECT ce.subject_id,
    max(ce.stay_id) AS stay_id,
    ce.charttime,
    max(
        CASE
            WHEN ce.itemid = 224688::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS respiratory_rate_set,
    max(
        CASE
            WHEN ce.itemid = 224690::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS respiratory_rate_total,
    max(
        CASE
            WHEN ce.itemid = 224689::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS respiratory_rate_spontaneous,
    max(
        CASE
            WHEN ce.itemid = 224687::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS minute_volume,
    max(
        CASE
            WHEN ce.itemid = 224684::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS tidal_volume_set,
    max(
        CASE
            WHEN ce.itemid = 224685::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS tidal_volume_observed,
    max(
        CASE
            WHEN ce.itemid = 224686::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS tidal_volume_spontaneous,
    max(
        CASE
            WHEN ce.itemid = 224696::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS plateau_pressure,
    max(
        CASE
            WHEN ce.itemid = ANY (ARRAY[220339::numeric, 224700::numeric]) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS peep,
    max(
        CASE
            WHEN ce.itemid = 223835::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS fio2,
    max(
        CASE
            WHEN ce.itemid = 224691::numeric THEN ce.valuenum
            ELSE NULL::double precision
        END) AS flow_rate,
    max(
        CASE
            WHEN ce.itemid = 223849::numeric THEN ce.value
            ELSE NULL::text
        END) AS ventilator_mode,
    max(
        CASE
            WHEN ce.itemid = 229314::numeric THEN ce.value
            ELSE NULL::text
        END) AS ventilator_mode_hamilton,
    max(
        CASE
            WHEN ce.itemid = 223848::numeric THEN ce.value
            ELSE NULL::text
        END) AS ventilator_type
   FROM ce
  GROUP BY ce.subject_id, ce.charttime
WITH DATA;