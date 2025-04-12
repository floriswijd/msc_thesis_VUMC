-- public.hourly_state_with_vent_settings_and_labval source

CREATE MATERIALIZED VIEW public.hourly_state_with_vent_settings_and_labval
TABLESPACE pg_default
AS WITH labs AS (
         SELECT le.subject_id,
            le.charttime::timestamp without time zone AS lab_time,
            max(
                CASE
                    WHEN le.itemid = 50821 THEN le.valuenum
                    ELSE NULL::integer
                END) AS po2,
            max(
                CASE
                    WHEN le.itemid = 50818 THEN le.valuenum
                    ELSE NULL::integer
                END) AS pco2,
            max(
                CASE
                    WHEN le.itemid = 50820 THEN le.valuenum
                    ELSE NULL::integer
                END) AS ph,
            max(
                CASE
                    WHEN le.itemid = 50802 THEN le.valuenum
                    ELSE NULL::integer
                END) AS base_excess,
            max(
                CASE
                    WHEN le.itemid = 50803 THEN le.valuenum
                    ELSE NULL::integer
                END) AS calc_bicarb,
            max(
                CASE
                    WHEN le.itemid = 50813 THEN le.valuenum
                    ELSE NULL::integer
                END) AS lactate,
            max(
                CASE
                    WHEN le.itemid = 50821 THEN le.flag
                    ELSE NULL::text
                END) AS flag_po2,
            max(
                CASE
                    WHEN le.itemid = 50818 THEN le.flag
                    ELSE NULL::text
                END) AS flag_pco2,
            max(
                CASE
                    WHEN le.itemid = 50820 THEN le.flag
                    ELSE NULL::text
                END) AS flag_ph,
            max(
                CASE
                    WHEN le.itemid = 50802 THEN le.flag
                    ELSE NULL::text
                END) AS flag_base_excess,
            max(
                CASE
                    WHEN le.itemid = 50803 THEN le.flag
                    ELSE NULL::text
                END) AS flag_calc_bicarb,
            max(
                CASE
                    WHEN le.itemid = 50813 THEN le.flag
                    ELSE NULL::text
                END) AS flag_lactate
           FROM labevents le
          WHERE le.itemid = ANY (ARRAY[50821, 50818, 50820, 50802, 50803, 50813])
          GROUP BY le.subject_id, (le.charttime::timestamp without time zone)
        ), stay_map AS (
         SELECT icustays.stay_id,
            icustays.subject_id
           FROM icustays
        ), vent AS (
         SELECT vs.stay_id,
            vs.charttime,
            vs.respiratory_rate_set,
            vs.respiratory_rate_total,
            vs.respiratory_rate_spontaneous,
            vs.minute_volume,
            vs.tidal_volume_set,
            vs.tidal_volume_observed,
            vs.tidal_volume_spontaneous,
            vs.plateau_pressure,
            vs.peep,
            vs.fio2,
            vs.flow_rate,
            vs.ventilator_mode,
            vs.ventilator_mode_hamilton,
            vs.ventilator_type
           FROM ventilator_settings_mat vs
        ), addl AS (
         SELECT ce.stay_id,
            ce.charttime::timestamp without time zone AS charttime,
            max(
                CASE
                    WHEN ce.itemid = 223872::numeric THEN ce.valuenum
                    ELSE NULL::real
                END) AS inspired_gas_temp,
            max(
                CASE
                    WHEN ce.itemid = 227517::numeric THEN ce.value
                    ELSE NULL::text
                END) AS humidification,
            max(
                CASE
                    WHEN ce.itemid = 229244::numeric THEN ce.value
                    ELSE NULL::text
                END) AS humidifier_water_changed,
            max(
                CASE
                    WHEN ce.itemid = 229245::numeric THEN ce.valuenum
                    ELSE NULL::real
                END) AS humidifier_water_fill_level
           FROM chartevents ce
          WHERE (ce.itemid = ANY (ARRAY[223872::numeric, 227517::numeric, 229244::numeric, 229245::numeric])) AND ce.value IS NOT NULL AND ce.stay_id IS NOT NULL
          GROUP BY ce.stay_id, (ce.charttime::timestamp without time zone)
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
    avg(vent.respiratory_rate_set) AS respiratory_rate_set,
    avg(vent.respiratory_rate_total) AS respiratory_rate_total,
    avg(vent.respiratory_rate_spontaneous) AS respiratory_rate_spontaneous,
    avg(vent.minute_volume) AS minute_volume,
    avg(vent.tidal_volume_set) AS tidal_volume_set,
    avg(vent.tidal_volume_observed) AS tidal_volume_observed,
    avg(vent.tidal_volume_spontaneous) AS tidal_volume_spontaneous,
    avg(vent.plateau_pressure) AS plateau_pressure,
    avg(vent.peep) AS peep,
    avg(vent.fio2) AS fio2,
    avg(vent.flow_rate) AS flow_rate,
    max(vent.ventilator_mode) AS ventilator_mode,
    max(vent.ventilator_mode_hamilton) AS ventilator_mode_hamilton,
    max(vent.ventilator_type) AS ventilator_type,
    avg(addl.inspired_gas_temp) AS inspired_gas_temp,
    max(addl.humidification) AS humidification,
    max(addl.humidifier_water_changed) AS humidifier_water_changed,
    avg(addl.humidifier_water_fill_level) AS humidifier_water_fill_level,
    avg(l.po2) AS po2,
    avg(l.pco2) AS pco2,
    avg(l.ph) AS ph,
    avg(l.base_excess) AS base_excess,
    avg(l.calc_bicarb) AS calc_bicarb,
    avg(l.lactate) AS lactate,
    max(l.flag_po2) AS flag_po2,
    max(l.flag_pco2) AS flag_pco2,
    max(l.flag_ph) AS flag_ph,
    max(l.flag_base_excess) AS flag_base_excess,
    max(l.flag_calc_bicarb) AS flag_calc_bicarb,
    max(l.flag_lactate) AS flag_lactate
   FROM hourly_state_mat hs
     LEFT JOIN vent ON hs.stay_id = vent.stay_id AND vent.charttime::timestamp without time zone >= (hs.hour_ts - '00:30:00'::interval) AND vent.charttime::timestamp without time zone <= (hs.hour_ts + '00:30:00'::interval)
     LEFT JOIN addl ON hs.stay_id = addl.stay_id AND addl.charttime >= (hs.hour_ts - '00:30:00'::interval) AND addl.charttime <= (hs.hour_ts + '00:30:00'::interval)
     LEFT JOIN stay_map sm ON hs.stay_id::numeric = sm.stay_id
     LEFT JOIN labs l ON sm.subject_id = l.subject_id::numeric AND l.lab_time >= (hs.hour_ts - '00:30:00'::interval) AND l.lab_time <= (hs.hour_ts + '00:30:00'::interval)
  GROUP BY hs.stay_id, hs.hour_ts, hs.heart_rate, hs.spo2, hs.resp_rate, hs.temperature, hs.sbp, hs.dbp, hs.o2_flow, hs.o2_flow_additional, hs.o2_delivery_device_1, hs.o2_delivery_device_2, hs.o2_delivery_device_3, hs.o2_delivery_device_4
WITH DATA;