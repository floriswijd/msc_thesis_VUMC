-- public.hourly_state_with_vent_outcome_2 source

CREATE MATERIALIZED VIEW public.hourly_state_with_vent_outcome_2
TABLESPACE pg_default
AS WITH outcome_raw AS (
         SELECT t1.stay_id,
                CASE
                    WHEN a.hospital_expire_flag = 1 THEN 'Death'::text
                    WHEN (EXISTS ( SELECT 1
                       FROM treatment_ventilation_mat t2
                      WHERE t2.stay_id = t1.stay_id AND t2.ventilation_status = 'InvasiveVent'::text AND t2.starttime::timestamp without time zone >= t1.endtime::timestamp without time zone AND t2.starttime::timestamp without time zone <= (t1.endtime::timestamp without time zone + '24:00:00'::interval))) THEN 'InvasiveVent'::text
                    ELSE 'Success'::text
                END AS outcome_label
           FROM treatment_ventilation_mat t1
             LEFT JOIN icustays i ON t1.stay_id::numeric = i.stay_id
             LEFT JOIN admissions a ON i.hadm_id = a.hadm_id::numeric
          WHERE t1.ventilation_status = 'HFNC'::text
        ), outcome_agg AS (
         SELECT outcome_raw.stay_id,
                CASE
                    WHEN max(
                    CASE outcome_raw.outcome_label
                        WHEN 'Death'::text THEN 3
                        WHEN 'InvasiveVent'::text THEN 2
                        ELSE 1
                    END) = 3 THEN 'Death'::text
                    WHEN max(
                    CASE outcome_raw.outcome_label
                        WHEN 'Death'::text THEN 3
                        WHEN 'InvasiveVent'::text THEN 2
                        ELSE 1
                    END) = 2 THEN 'InvasiveVent'::text
                    ELSE 'Success'::text
                END AS outcome_label
           FROM outcome_raw
          GROUP BY outcome_raw.stay_id
        )
 SELECT hs.stay_id,
    hs.hour_ts,
    hs.heart_rate,
    hs.spo2,
    hs.resp_rate,
    hs.temperature,
    hs.sbp,
    hs.dbp,
    hs.o2_flow,
    hs.o2_flow_additional,
    hs.o2_delivery_device_1,
    hs.o2_delivery_device_2,
    hs.o2_delivery_device_3,
    hs.o2_delivery_device_4,
    hs.respiratory_rate_set,
    hs.respiratory_rate_total,
    hs.respiratory_rate_spontaneous,
    hs.minute_volume,
    hs.tidal_volume_set,
    hs.tidal_volume_observed,
    hs.tidal_volume_spontaneous,
    hs.plateau_pressure,
    hs.peep,
    hs.fio2,
    hs.flow_rate,
    hs.ventilator_mode,
    hs.ventilator_mode_hamilton,
    hs.ventilator_type,
    hs.inspired_gas_temp,
    hs.humidification,
    hs.humidifier_water_changed,
    hs.humidifier_water_fill_level,
    hs.po2,
    hs.pco2,
    hs.ph,
    hs.base_excess,
    hs.calc_bicarb,
    hs.lactate,
    hs.flag_po2,
    hs.flag_pco2,
    hs.flag_ph,
    hs.flag_base_excess,
    hs.flag_calc_bicarb,
    hs.flag_lactate,
    sm.subject_id,
    o.outcome_label
   FROM hourly_state_with_vent_settings_and_labval hs
     LEFT JOIN icustays sm ON hs.stay_id::numeric = sm.stay_id
     LEFT JOIN outcome_agg o ON hs.stay_id = o.stay_id
WITH DATA;