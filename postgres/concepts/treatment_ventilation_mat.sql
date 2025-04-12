-- public.treatment_ventilation_mat source

CREATE MATERIALIZED VIEW public.treatment_ventilation_mat
TABLESPACE pg_default
AS WITH tm AS (
         SELECT ventilator_settings.stay_id,
            ventilator_settings.charttime
           FROM ventilator_settings_mat ventilator_settings
        UNION
         SELECT oxygen_delivery.stay_id,
            oxygen_delivery.charttime
           FROM oxygen_delivery_mat oxygen_delivery
        ), vs AS (
         SELECT tm.stay_id,
            tm.charttime,
            od.o2_delivery_device_1,
            COALESCE(vs.ventilator_mode, vs.ventilator_mode_hamilton) AS vent_mode,
                CASE
                    WHEN od.o2_delivery_device_1 = ANY (ARRAY['Tracheostomy tube'::text, 'Trach mask '::text]) THEN 'Tracheostomy'::text
                    WHEN od.o2_delivery_device_1 = 'Endotracheal tube'::text OR (vs.ventilator_mode = ANY (ARRAY['(S) CMV'::text, 'APRV'::text, 'APRV/Biphasic+ApnPress'::text, 'APRV/Biphasic+ApnVol'::text, 'APV (cmv)'::text, 'Ambient'::text, 'Apnea Ventilation'::text, 'CMV'::text, 'CMV/ASSIST'::text, 'CMV/ASSIST/AutoFlow'::text, 'CMV/AutoFlow'::text, 'CPAP/PPS'::text, 'CPAP/PSV'::text, 'CPAP/PSV+Apn TCPL'::text, 'CPAP/PSV+ApnPres'::text, 'CPAP/PSV+ApnVol'::text, 'MMV'::text, 'MMV/AutoFlow'::text, 'MMV/PSV'::text, 'MMV/PSV/AutoFlow'::text, 'P-CMV'::text, 'PCV+'::text, 'PCV+/PSV'::text, 'PCV+Assist'::text, 'PRES/AC'::text, 'PRVC/AC'::text, 'PRVC/SIMV'::text, 'PSV/SBT'::text, 'SIMV'::text, 'SIMV/AutoFlow'::text, 'SIMV/PRES'::text, 'SIMV/PSV'::text, 'SIMV/PSV/AutoFlow'::text, 'SIMV/VOL'::text, 'SYNCHRON MASTER'::text, 'SYNCHRON SLAVE'::text, 'VOL/AC'::text])) OR (vs.ventilator_mode_hamilton = ANY (ARRAY['APRV'::text, 'APV (cmv)'::text, 'Ambient'::text, '(S) CMV'::text, 'P-CMV'::text, 'SIMV'::text, 'APV (simv)'::text, 'P-SIMV'::text, 'VS'::text, 'ASV'::text])) THEN 'InvasiveVent'::text
                    WHEN (od.o2_delivery_device_1 = ANY (ARRAY['Bipap mask '::text, 'CPAP mask '::text])) OR (vs.ventilator_mode_hamilton = ANY (ARRAY['DuoPaP'::text, 'NIV'::text, 'NIV-ST'::text])) THEN 'NonInvasiveVent'::text
                    WHEN od.o2_delivery_device_1 = 'High flow nasal cannula'::text THEN 'HFNC'::text
                    WHEN od.o2_delivery_device_1 = ANY (ARRAY['Non-rebreather'::text, 'Face tent'::text, 'Aerosol-cool'::text, 'Venti mask '::text, 'Medium conc mask '::text, 'Ultrasonic neb'::text, 'Vapomist'::text, 'Oxymizer'::text, 'High flow neb'::text, 'Nasal cannula'::text]) THEN 'SupplementalOxygen'::text
                    WHEN od.o2_delivery_device_1 = 'None'::text THEN 'None'::text
                    ELSE NULL::text
                END AS ventilation_status
           FROM tm
             LEFT JOIN ventilator_settings_mat vs ON tm.stay_id = vs.stay_id AND tm.charttime::text = vs.charttime::text
             LEFT JOIN oxygen_delivery_mat od ON tm.stay_id = od.stay_id AND tm.charttime::text = od.charttime::text
        ), vd0 AS (
         SELECT vs.stay_id,
            vs.charttime,
            lag(vs.charttime, 1) OVER (PARTITION BY vs.stay_id, vs.ventilation_status ORDER BY vs.charttime) AS charttime_lag,
            lead(vs.charttime, 1) OVER w AS charttime_lead,
            vs.ventilation_status,
            lag(vs.ventilation_status, 1) OVER w AS ventilation_status_lag
           FROM vs
          WHERE vs.ventilation_status IS NOT NULL
          WINDOW w AS (PARTITION BY vs.stay_id ORDER BY vs.charttime)
        ), vd1 AS (
         SELECT vd0.stay_id,
            vd0.charttime,
            vd0.charttime_lag,
            vd0.charttime_lead,
            vd0.ventilation_status,
            EXTRACT(epoch FROM vd0.charttime::timestamp without time zone - vd0.charttime_lag::timestamp without time zone) / 3600::numeric AS ventduration,
                CASE
                    WHEN vd0.ventilation_status_lag IS NULL THEN 1
                    WHEN (EXTRACT(epoch FROM vd0.charttime::timestamp without time zone - vd0.charttime_lag::timestamp without time zone) / 3600::numeric) >= 14::numeric THEN 1
                    WHEN vd0.ventilation_status_lag <> vd0.ventilation_status THEN 1
                    ELSE 0
                END AS new_ventilation_event
           FROM vd0
        ), vd2 AS (
         SELECT vd1.stay_id,
            vd1.charttime,
            vd1.charttime_lead,
            vd1.ventilation_status,
            vd1.ventduration,
            vd1.new_ventilation_event,
            sum(vd1.new_ventilation_event) OVER (PARTITION BY vd1.stay_id ORDER BY vd1.charttime) AS vent_seq
           FROM vd1
        )
 SELECT vd2.stay_id,
    min(vd2.charttime::text) AS starttime,
    max(
        CASE
            WHEN vd2.charttime_lead IS NULL OR (EXTRACT(epoch FROM vd2.charttime_lead::timestamp without time zone - vd2.charttime::timestamp without time zone) / 3600::numeric) >= 14::numeric THEN vd2.charttime
            ELSE vd2.charttime_lead
        END::text) AS endtime,
    max(vd2.ventilation_status) AS ventilation_status
   FROM vd2
  GROUP BY vd2.stay_id, vd2.vent_seq
 HAVING min(vd2.charttime::text) <> max(vd2.charttime::text)
WITH DATA;