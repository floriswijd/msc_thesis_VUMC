SELECT ce.*,
       di."label",
       di."category",
       di.param_type 
FROM chartevents ce
LEFT JOIN d_items di ON ce.itemid = di.itemid
WHERE ce.stay_id IN (
    SELECT DISTINCT stay_id
    FROM chartevents
    WHERE itemid = 50816
)
ORDER BY ce.stay_id desc
limit 500;

