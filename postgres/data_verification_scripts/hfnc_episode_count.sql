-- Query summary statistics from the materialized view
SELECT 
    COUNT(*) AS total_hfnc_episodes,
    MIN(duration_hours) AS min_duration_hours,
    MAX(duration_hours) AS max_duration_hours,
    AVG(duration_hours) AS avg_duration_hours,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_hours) AS median_duration_hours
FROM hfnc_episodes_mat;

-- Identify ICU stays with multiple HFNC episodes
SELECT 
    stay_id,
    COUNT(*) AS num_hfnc_episodes
FROM hfnc_episodes_mat
GROUP BY stay_id
HAVING COUNT(*) > 1
ORDER BY num_hfnc_episodes DESC;