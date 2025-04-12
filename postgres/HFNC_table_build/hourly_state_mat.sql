CREATE MATERIALIZED VIEW public.hourly_state_mat
TABLESPACE pg_default
AS WITH vitals AS (
         SELECT vitalsign_mat.subject_id,
            vitalsign_mat.stay_id,
            vitalsign_mat.charttime,
            vitalsign_mat.heart_rate,
            vitalsign_mat.spo2,
            vitalsign_mat.resp_rate,
            vitalsign_mat.temperature,
            vitalsign_mat.sbp,
            vitalsign_mat.dbp
           FROM vitalsign_mat
        ), oxy AS (
         SELECT oxygen_delivery_mat.subject_id,
            oxygen_delivery_mat.stay_id,
            oxygen_delivery_mat.charttime,
            oxygen_delivery_mat.o2_flow,
            oxygen_delivery_mat.o2_flow_additional,
            oxygen_delivery_mat.o2_delivery_device_1,
            oxygen_delivery_mat.o2_delivery_device_2,
            oxygen_delivery_mat.o2_delivery_device_3,
            oxygen_delivery_mat.o2_delivery_device_4
           FROM oxygen_delivery_mat
        )
 SELECT h.stay_id,
    h."timestamp" AS hour_ts,
    avg(v.heart_rate) AS heart_rate,
    avg(v.spo2) AS spo2,
    avg(v.resp_rate) AS resp_rate,
    avg(v.temperature) AS temperature,
    avg(v.sbp) AS sbp,
    avg(v.dbp) AS dbp,
    avg(oxy.o2_flow) AS o2_flow,
    avg(oxy.o2_flow_additional) AS o2_flow_additional,
    max(oxy.o2_delivery_device_1) AS o2_delivery_device_1,
    max(oxy.o2_delivery_device_2) AS o2_delivery_device_2,
    max(oxy.o2_delivery_device_3) AS o2_delivery_device_3,
    max(oxy.o2_delivery_device_4) AS o2_delivery_device_4
   FROM hfnc_hourly_intervals_mat h
     LEFT JOIN vitals v ON v.stay_id = h.stay_id AND v.charttime::timestamp without time zone >= (h."timestamp" - '00:30:00'::interval) AND v.charttime::timestamp without time zone <= (h."timestamp" + '00:30:00'::interval)
     LEFT JOIN oxy ON oxy.stay_id = h.stay_id AND oxy.charttime::timestamp without time zone >= (h."timestamp" - '00:30:00'::interval) AND oxy.charttime::timestamp without time zone <= (h."timestamp" + '00:30:00'::interval)
  GROUP BY h.stay_id, h."timestamp"
WITH DATA;