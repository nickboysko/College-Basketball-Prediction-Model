@echo off
echo ================================================================================
echo COLLEGE BASKETBALL SPREAD PREDICTION MODEL
echo ================================================================================
echo.

echo Step 1: Merging game data with team statistics...
echo.
python merge_data.py
if %errorlevel% neq 0 (
    echo ERROR: merge_data.py failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ================================================================================
echo.
echo Step 2: Training models and evaluating performance...
echo.
python train_models.py
if %errorlevel% neq 0 (
    echo ERROR: train_models.py failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ================================================================================
echo ALL DONE!
echo ================================================================================
echo.
echo Output files created:
echo   - training_data.csv
echo   - testing_data.csv
echo   - predictions_with_best_model.csv
echo.
echo Check the predictions_with_best_model.csv file to see individual game predictions!
echo.
pause
