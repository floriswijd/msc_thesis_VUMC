-- public.hourly_state_with_vent_settings_and_addl_mat source

CREATE MATERIALIZED VIEW public.hourly_state_with_vent_settings_and_addl_mat
TABLESPACE pg_default
AS WITH vent AS (
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
    avg(addl.humidifier_water_fill_level) AS humidifier_water_fill_level
   FROM hourly_state_mat hs
     LEFT JOIN vent ON hs.stay_id = vent.stay_id AND vent.charttime::timestamp without time zone >= (hs.hour_ts - '00:30:00'::interval) AND vent.charttime::timestamp without time zone <= (hs.hour_ts + '00:30:00'::interval)
     LEFT JOIN addl ON hs.stay_id = addl.stay_id AND addl.charttime >= (hs.hour_ts - '00:30:00'::interval) AND addl.charttime <= (hs.hour_ts + '00:30:00'::interval)
  GROUP BY hs.stay_id, hs.hour_ts, hs.heart_rate, hs.spo2, hs.resp_rate, hs.temperature, hs.sbp, hs.dbp, hs.o2_flow, hs.o2_flow_additional, hs.o2_delivery_device_1, hs.o2_delivery_device_2, hs.o2_delivery_device_3, hs.o2_delivery_device_4
WITH DATA;