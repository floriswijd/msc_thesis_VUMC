-- public.hfnc_episodes_mat source

CREATE MATERIALIZED VIEW public.hfnc_episodes_mat
TABLESPACE pg_default
AS SELECT treatment_ventilation.stay_id,
    treatment_ventilation.starttime,
    treatment_ventilation.endtime,
    treatment_ventilation.ventilation_status,
    EXTRACT(epoch FROM treatment_ventilation.endtime::timestamp without time zone - treatment_ventilation.starttime::timestamp without time zone) / 3600::numeric AS duration_hours
   FROM treatment_ventilation
  WHERE treatment_ventilation.ventilation_status = 'HFNC'::text
WITH DATA;

-- View indexes:
CREATE INDEX idx_hfnc_stay ON public.hfnc_episodes_mat USING btree (stay_id);