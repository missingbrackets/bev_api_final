"""Task entry points for data fetching and peril modelling."""

from typing import List, Union
from datetime import date, datetime, timezone, timedelta
import re

import pandas as pd
import polars as pl
import hx

from algorithms.parameters.parameters import get_parameters

from algorithms.tasks_section.cat.sql_extract import cat_sql_extract
from algorithms.tasks_section.cat.cat_processing import camel_to_snake, cat_map_air_data, cat_lead_time, cat_time_to_event, calculate_event_impact, identify_pml_contributors, calculate_loss_windows, calculate_cat_losses_with_recovery, pml_events_join, pivot_cat_losses
from algorithms.tasks_section.prequel import db_query
from algorithms.tasks_section.adverse_weather import (
    bev_task_batches_threaded,
    normalise_daily_results,
    normalise_expanding_results,
    build_failed_warnings,
    ENDPOINT_MAX_COMBINATIONS,
    _CD_CONCURRENCY,
    _CD_BATCH_PAUSE_SECONDS,
    _DEFAULT_BASE_URL,
)
from algorithms.algorithm_utilities import pd_df_from_hx_list, get_nat_cat_perils

from algorithms.tasks_section.database import fetch_database

@hx.task
def fetch_prequel_data(hxd, progress):
    """
    Fetches Prequel data from the database using the Program ID and updates the hxd object with the relevant details.

    Parameters:
        hxd (object): The hxd object containing input fields and exposure data.
    
    Usage:
        This function retrieves program-related data such as client name, currency, inception and expiry dates, and more 
        by querying multiple database views using the Program ID. The retrieved information is used to populate the 
        `hxd.inputs` and `hxd.umcc_template` fields. If no data is found or if the Program ID is missing, the function logs 
        a failure message in `hxd.fetch_notes_after_fetch`.
    """


    # hxd.show_after_fetch_notes = True

    # if not hxd.policy_overview.prequel_info.pq_id:
    #     # hxd.fetch_notes_after_fetch = "Fetch Status: **FAILED** ??\n* No Program ID Entered"
    #     return

    program_id = hxd.policy_overview.prequel_info.pq_id
    layer_query = (
        f"""
        WITH programs AS(
        SELECT  
            p.[ProgramID]
                ,p.[Program_Name]
                ,p.[Inception]
                ,p.[Expiry]
                ,p.[Is_Binder]
                ,p.[Is_Prior_Submit]
                ,p.[Is_Declaration]
                ,p.[Binder_ProgramID]
            FROM [HX].[PrequelInterface].[Programs_v] p
            WHERE p.[ProgramID] = {program_id}
        ),
        layers AS(
            SELECT DISTINCT 
                l.[ProgramID]
                ,l.[Mapping_Code] 
            FROM HX.PrequelInterface.Layer_Details_Excel_Seed_Inc_Decs_v l 
            WHERE l.[ProgramID] = {program_id}
        )
        
        SELECT 
            p.[ProgramID]
            ,p.[Program_Name]
            ,p.[Inception]
            ,p.[Expiry]
            ,p.[Is_Binder]
            ,p.[Is_Prior_Submit]
            ,p.[Is_Declaration]
            ,p.[Binder_ProgramID]
            ,l.[Mapping_Code]
        FROM programs p
        LEFT JOIN layers l ON p.ProgramID = l.ProgramID
 
        """
    )

    rate_query = """
        SELECT * FROM [HX].[JarvisInterface].[Fx_t]
        WHERE official_period IN (
            SELECT MAX(official_period) AS max_official_period
            FROM [HX].[JarvisInterface].[Fx_t]
        )
        AND to_currency = 'usd'
    """
    with fetch_database(hxd) as connection:
        # Execute queries
        df_layer_data = db_query(connection, layer_query, return_one=False)
        df_fx_rates = db_query(connection, rate_query, return_one=False)

    base_model_date = datetime.now(timezone.utc).date()
    
    if df_layer_data is None or df_layer_data.empty:
        hxd.policy_overview.prequel_info.notes_after_fetch  = "Fetch Status: **FAILED** ??\n* No data found for the given Program ID"
        #   model date - model date = today unless cat has been run
        if not hxd.policy_overview.prequel_info.model_date.calculated:
            hxd.policy_overview.prequel_info.model_date.calculated = base_model_date
                # Handle empty result set appropriately
    else:
        # Update hxd with the retrieved data
        base_model_date = datetime.now(timezone.utc).date()
        # Update hxd with the retrieved data 
        if df_layer_data.empty:
            
            if not hxd.policy_overview.prequel_info.model_date.calculated:
                hxd.policy_overview.prequel_info.model_date.calculated = base_model_date

        # Update hxd with the retrieved data 
        hxd.policy_overview.prequel_info.program_name.calculated = df_layer_data['Program_Name'].iloc[0]
        hxd.policy_overview.prequel_info.inception.calculated   = df_layer_data['Inception'].iloc[0]
        hxd.policy_overview.prequel_info.expiry.calculated      = df_layer_data['Expiry'].iloc[0]
        hxd.policy_overview.prequel_info.mapping_code      = df_layer_data['Mapping_Code'].iloc[0]
        
        #   model date - model date = today unless cat has been run
        if not hxd.policy_overview.prequel_info.model_date.calculated:
            hxd.policy_overview.prequel_info.model_date.calculated = min(base_model_date, df_layer_data['Inception'].iloc[0]) - timedelta(days=1)
        
        # hxd.hx_core.inception_date = df_layer_data['Inception'].iloc[0]
        # hxd.hx_core.expiry_date = df_layer_data['Expiry'].iloc[0]

        hx.meta.pas_references=[f"{program_id}"]
        
        hxd.policy_overview.prequel_info.notes_after_fetch = "Fetch Status: **COMPLETE** ??"

    if df_fx_rates is None or df_fx_rates.empty:
        hxd.fx_rates.notes_after_fetch = "Fetch Status: **FAILED** ??\n* No FX data was found"
        pass
    else:
        data_cols = get_parameters("calculations/fx_rates/fx_sql_dictionary.json")
         # Rename the SQL columns to match the hxd fields
        df_fx_rates.rename(columns=data_cols, inplace=True)
    
        # Update the hxd object with the fetched FX rate data
        hxd.fx_rates.assumptions = df_fx_rates[list(data_cols.values())].to_dict(orient="records")
    
        # Mark the fetch operation as complete
        hxd.fx_rates.notes_after_fetch = "Fetch Status: **COMPLETE** ??"
    return

@hx.task
def mourning_named_person_cross_join(hxd, progress):

    # plan: expand based on length of named perosns input
    # Once expanded, need to get if more rows than that exist per trigger
    # if yes then keep them
    # if no add to them 
    # if table exists then call rating.

    df_event_info_join = pd_df_from_hx_list(hxd.event_level.event_info)[[
            "index", 
    ]].rename(columns={
            "index": "event_index",
    })
    
    df_named_person_input = pd_df_from_hx_list(hxd.mourning.mourning_named_person_input)

    # list_existing = hxd.mourning.mourning_named_person_calculations

    # if len(list_existing) > 1:
    #     df_existing = pd_df_from_hx_list(list_existing)

    df_mortality = pd_df_cross_join_validate(
        df_left = df_event_info_join,
        df_right = df_named_person_input,
        # df_existing = df_existing,
        left_index="event_index",
        right_index="person_index"
    )
    
    hxd.mourning.mourning_named_person_task_output = df_mortality.fillna(0).to_dict(orient="records")
    return

@hx.task
def mourning_reset(hxd, progress):
    df =  pd_df_from_hx_list(hxd.mourning.mourning_named_person_task_output)
    blank_df =  pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
    blank_df['named_person'] = " " #This just avoids a task error that the named person field is not optional
    hxd.mourning.mourning_named_person_task_output = blank_df.to_dict(orient="records")
    return

@hx.task
def non_app_named_person_cross_join(hxd, progress):

    df_event_info_join = pd_df_from_hx_list(hxd.event_level.event_info)[[
            "index", 
    ]].rename(columns={
            "index": "event_index",
    })
    
    df_named_person_input = pd_df_from_hx_list(hxd.non_app.non_app_named_person_input)

    performer_only = len(df_named_person_input.loc[df_named_person_input["relationship"] != "Performer"].index) == 0
    named_person_extension = hxd.non_app.named_person_extension

    if performer_only and named_person_extension:
        df_named_person_input["join_index"] = 0
        df_default_relationships = hx.params.non_app_default_relationships
        df_default_relationships["join_index"] = 0
        df_default_relationships = df_default_relationships.merge(
            df_named_person_input[["join_index", "age", "name"]],
            on = "join_index",
            suffixes = ["_family", None]
        )
        df_default_relationships["age"] = df_default_relationships["age"] + df_default_relationships["age_gap"]
        df_default_relationships["name"] = df_default_relationships["name"] + " - " + df_default_relationships["name_family"] # tweak for empty name
        age_limits = get_parameters("calculations/validation/validation.json")["age_limits"]
        
        df_default_relationships['age']= df_default_relationships['age'].clip(lower=age_limits['gte'], upper=age_limits["lte"])
        
        df_named_person_input = pd.concat([df_named_person_input.drop(columns={"join_index"}), df_default_relationships.drop(columns={"age_gap", "join_index", "name_family"})]).reset_index(drop=True)
        df_named_person_input["person_index"] = df_named_person_input.index

    # list_existing = hxd.calculations.non_appearance.non_app_named_person_task_output

    # if len(list_existing) > 1:
    #     df_existing = pd_df_from_hx_list(list_existing)

    df_mortality = pd_df_cross_join_validate(
        df_left = df_event_info_join,
        df_right = df_named_person_input,
        # df_existing = df_existing,
        left_index="event_index",
        right_index="person_index"
    )

    #hx.errors.fatal(df_mortality['age'])

    hxd.calculations.non_appearance.non_app_named_person_task_output = df_mortality.fillna(0).to_dict(orient="records")
    
    return

@hx.task
def non_app_named_person_reset(hxd, progress):
    df =  pd_df_from_hx_list(hxd.calculations.non_appearance.non_app_named_person_task_output)
    blank_df =  pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
    hxd.calculations.non_appearance.non_app_named_person_task_output = blank_df.to_dict(orient="records")
    return

@hx.task
def mourning_ncr_cross_join(hxd, progress):

    df_event_info_join = pd_df_from_hx_list(hxd.event_level.event_info)[[
            "index", 
    ]].rename(columns={
            "index": "event_index",
    })

    df_ncr_input = pd_df_from_hx_list(hxd.mourning.mourning_ncr_input)

    # list_existing = hxd.calculations.non_appearance.non_app_named_person_task_output

    # if len(list_existing) > 1:
    #     df_existing = pd_df_from_hx_list(list_existing)

    df_mortality = pd_df_cross_join_validate(
        df_left = df_event_info_join,
        df_right = df_ncr_input,
        # df_existing = df_existing,
        left_index="event_index",
        right_index="ncr_index"
    )

    hxd.mourning.mourning_ncr_task_output = df_mortality.fillna(0).to_dict(orient="records")
    
    return

@hx.task
def mourning_ncr_reset(hxd, progress):
    df =  pd_df_from_hx_list(hxd.mourning.mourning_ncr_task_output)
    blank_df =  pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
    hxd.mourning.mourning_ncr_task_output = blank_df.to_dict(orient="records")
    return

# named person/group linked dropdown
# 

def pd_df_cross_join_validate(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_index: str,
    right_index: str,
    df_existing: pd.DataFrame = None
) -> pd.DataFrame:
    """Cross-join two DataFrames, add trigger_index, and validate/augment against existing.
    
    Parameters
    ----------
    df_left : pd.DataFrame
        Left DataFrame to cross-join.
    df_right : pd.DataFrame  
        Right DataFrame to cross-join.
    df_existing : pd.DataFrame
        Existing DataFrame to validate against and potentially augment.
    group_key : str or list
        Column name(s) defining groups for row count comparison.
    value_columns : list, optional
        Currently unused, kept for backward compatibility.
    fill_values : dict, optional
        Default values for non-key columns when adding new rows.
        
    Returns
    -------
    pd.DataFrame
        df_existing with missing rows appended (if any). Never removes rows.
    """
    
    # Cross-join the two DataFrames
    df_expected = df_left.merge(df_right, how='cross')

    right_row_count = len(df_right)
    
    # Add trigger_index column (1-indexed row number)
    df_expected['trigger_index'] = df_expected[left_index] * right_row_count + df_expected[right_index]
    
    return df_expected


        # # Get all columns from expected (will be used as key columns for comparison)
        # key_cols = df_expected.columns.tolist()
        
        # # Vectorized approach: find all missing rows using anti-join
        # # Get group counts for comparison
        # expected_counts = df_expected.groupby(group_key).size().reset_index(name='expected_count')
        # existing_counts = df_existing.groupby(group_key).size().reset_index(name='existing_count')
        
        # # Merge counts to identify groups with fewer rows in existing
        # count_comparison = expected_counts.merge(existing_counts, on=group_key, how='left')
        # count_comparison['existing_count'] = count_comparison['existing_count'].fillna(0).astype(int)
        
        # # Print group-level differences for diagnostics
        # groups_needing_rows = count_comparison[count_comparison['existing_count'] < count_comparison['expected_count']]
        # if not groups_needing_rows.empty:
        #     print(f"Groups with missing rows: {len(groups_needing_rows)}")
        
        # # Find missing rows using anti-join on all key columns
        # # Merge with indicator to identify rows only in expected
        # df_missing = df_expected.merge(
        #     df_existing[key_cols] if set(key_cols).issubset(df_existing.columns) else df_existing,
        #     on=[col for col in key_cols if col in df_existing.columns],
        #     how='left',
        #     indicator=True
        # )
        # df_missing = df_missing[df_missing['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        # # If there are missing rows, prepare them for concatenation
        # if not df_missing.empty:
            
        #     # Add non-key columns with fill values or NaN
        #     for col in df_existing.columns:
        #         if col not in df_missing.columns:
        #             if fill_values and col in fill_values:
        #                 df_missing[col] = fill_values[col]
        #             else:
        #                 df_missing[col] = pd.NA
            
        #     # Ensure column order matches existing
        #     df_missing = df_missing[df_existing.columns]
            
        #     # Append to existing
        #     df_result = pd.concat([df_existing, df_missing], ignore_index=True)
        #     print(f"Added {len(df_missing)} missing rows")
            
        # else:
        #     print("No missing rows to add")
        #     return df_existing

            # df_table_1, df_table_2:
    # cross_jon: df_table_1.merge(df_table_2, how = "cross")
    # check row_count: 
    # check row_count by grouping: 
    # grouping:
    #   non-app: event x name combination
    #   named_person: event x named_person combination
    #   ncr: event x ncr
    # if row_count by groping less - add empty row with that groupoing
    # if row_count by grouping more - do nothing
    # inputs: 
    #   non-app
    #   named_person
    #   ncr: 

@hx.task
def run_rating(hxd, progress):
    hxd.event_level.summary_info.rating_run = False
    perils = hxd.policy_overview.coverage_overview.perils
    try:
        if perils.nat_cat.include:
            cat_run(hxd, progress)
    except Exception as e:
        hx.errors.fatal(f"Cat model failed: {str(e)},, please check inputs and try again")
    
    try:
        if perils.adverse_weather.include:
            adverse_weather_rating(hxd, progress)
    except Exception as e:
        hx.errors.fatal(f"Adverse weather rating failed: {str(e)}, please check inputs and try again")

    hxd.event_level.summary_info.rating_run = True
    return

@hx.task
def cat(hxd, progress):
    cat_run(hxd, progress)
    return

# @hx.task
def cat_run(hxd, progress):
    """Run CAT extraction task and attach outputs to the context."""
    df_locations = pl.from_pandas(
        pd_df_from_hx_list(hxd.event_level.event_info)
    )

    nat_cat_perils = get_nat_cat_perils()
    nat_cat_peril_columns = [f"{peril}_{suffix}" for suffix in ("xs", "limit") for peril in nat_cat_perils]

    df_locations = df_locations.rename(
        {
            c: re.sub(r'^(location/|risk_factors/)', '', c)
            for c in df_locations.columns
            if c.startswith('location/') or c.startswith('risk_factors/')
        }
    ).select(
            [
                "index",
                "currency", 
                "driver", 
                "event_xs_usd", 
                "event_duration", 
                "event_end_date", 
                "event_limit_usd", 
                "event_start_date",
                "event_venue_name",
                "layer_id", 
                "length",
                "modelled_amount", 
                "time_until",
                "country",
                "area", 
                "city",
                "event_subset",
                "event_type",
                "venue_type",
                "fx"
            ] + nat_cat_peril_columns
        ).with_columns(
            [pl.col(condition).cast(pl.Float64) for condition in nat_cat_peril_columns + ["fx"]] 
        )

    # Assign air_model_id and peril
    df_air_mapping = pl.from_pandas(hx.params.air_mapping).with_columns(
        pl.when(
            pl.col("country").is_in(["US and Canada", "North America"])
        ).then(
            pl.lit("United States")
        ).otherwise(
            pl.col("country")
        ).alias("country")
    ).select(
        "country", "state", "modelled_min_dr", "air_model_id", "peril"
    ).unique()

    df_air_countries = pl.from_pandas(
        hx.params.air_countries
    ).select(
        "legacy_country", "model_region"
    )

    # Retreive flexible revenue proportion
    df_params_flex_revenue = pl.from_pandas(
        hx.params.adverse_weather_thresholds[
            ["event_type", "event_subset", "venue_type", "region", "proportion_flexible_revenue"]
        ]
    )

    # split table between rows that have 
    df_params_flex_revenue_country = df_params_flex_revenue.filter(
        pl.col("region") != ""
    )

    df_params_flex_revenue_no_country = df_params_flex_revenue.filter(
        pl.col("region") == ""
    ).select( ["event_type", "event_subset", "venue_type", "proportion_flexible_revenue"])

    # Assign location ids based on country, area and city
    df_location_id = pl.from_pandas(hx.params.area_city_location_id_mapping)

    df_locations = df_locations.join(
        df_location_id,
        on=["country", "area", "city"],
        how="left" 
    ).join(
        df_air_countries,
        how="left",
        left_on=["country"],
        right_on=["legacy_country"]
    ).join(
        df_params_flex_revenue_country,
        left_on=["event_type", "event_subset", "venue_type", "model_region"],
        right_on=["event_type", "event_subset", "venue_type", "region",],
        how="left"
    ).join(
        df_params_flex_revenue_no_country,
        left_on=["event_type", "event_subset", "venue_type",],
        right_on=["event_type", "event_subset", "venue_type", ],
        how="left"
    ).with_columns(
        pl.coalesce("proportion_flexible_revenue", "proportion_flexible_revenue_right").alias("proportion_flexible_revenue")
    ).drop(
        "proportion_flexible_revenue_right"
    ).join(
        df_air_mapping,
        left_on = ["area"],
        right_on = ["state"],
        how="left"
    )

    # update for non-match locations on state
    df_locations = df_locations.update(
        df_air_mapping,
        left_on = ["model_region"],
        right_on = ["country"]
    )

    # update for non-match locations on state
    df_locations = df_locations.update(
        df_air_mapping,
        left_on = ["country"],
        right_on = ["state"]
    )

    df_cat_events, df_pml_events = cat_sql_extract(df_locations)
    if df_cat_events.height > 0:
        # Convert column names from camelCase to snake_case
        df_cat_events = (
            df_cat_events
            .with_columns(
                pl.col("LocationID").cast(pl.Int64),
                pl.col("EventID").cast(pl.Int64)
            )
        ).rename({
            col: camel_to_snake(col)
            for col in df_cat_events.columns
        })

        
        df_pml_events = df_pml_events.with_columns(
                pl.col("EventId").cast(pl.Int64)
            ).rename({
                col: camel_to_snake(col).replace("__", "_").replace(".", "_")
                for col in df_pml_events.columns
            })

        df_cat_events = cat_map_air_data(df_cat_events, df_air_mapping)

        df_natcat_limits = df_locations.select(
            [
                "index",
                "location_id",
                "modelled_amount",
                "fx",
            ] + nat_cat_peril_columns
        ).unique()

        df_model_constants = pl.from_pandas(hx.params.cat_model_constants)
        df_nat_cat_assumptions = pl.from_pandas(hx.params.nat_cat_assumptions)
        df_recovery = pl.from_pandas(hx.params.nat_cat_recovery_row)
        
        df_events = df_locations.select(
            "index",
            "location_id",
            "event_start_date",
            "event_end_date",
            "modelled_amount",
            "venue_type",
            "proportion_flexible_revenue",
            "event_xs_usd",
            "event_limit_usd",
        ).unique().with_columns(
            pl.col("event_start_date").dt.ordinal_day().alias("event_start_day"),
            pl.col("event_end_date").dt.ordinal_day().alias("event_end_day"),
            pl.col("event_xs_usd").cast(pl.Float64).fill_null(0),
            pl.col("event_limit_usd").cast(pl.Float64).fill_null(float("inf"))
        )
        # if agg limit is none, then exposure, else if layer_limit > limited exposure then limited event exposure else laeyr limit
        agg_limit = hxd.policy_overview.terms.aggregate_limits.agg_limit_usd_capped
        if agg_limit is None:
            hx.errors.validation("Aggregate limit (USD capped) is required for CAT calculations")
            agg_limit = float("inf")
        
        agg_deductible = hxd.policy_overview.terms.aggregate_limits.agg_deductible_usd.selected
        if agg_deductible is None:
            agg_deductible = 0

        df_cat_events = cat_map_air_data(df_cat_events, df_air_mapping)
        df_cat_events = cat_lead_time(df_cat_events, df_nat_cat_assumptions, df_model_constants)
        df_cat_events = cat_time_to_event(df_cat_events, df_events)
        df_cat_events = calculate_event_impact(df_cat_events, df_events)
        df_cat_events = identify_pml_contributors(df_cat_events, df_pml_events)
        df_cat_events = calculate_loss_windows(df_cat_events, df_events, df_recovery)
        df_cat_events, df_year_loss_summary, df_index_summary = calculate_cat_losses_with_recovery(
            df_cat_events,
            df_events,
            df_natcat_limits,
            df_recovery,
            agg_limit,
            agg_deductible,
            1
        )

        df_pml_events, df_peril_seasonal = pml_events_join(df_cat_events, df_pml_events)
        hxd.calculations.pml.pml_events = df_pml_events.to_dicts()
        hxd.calculations.pml.pml_seasonality = df_peril_seasonal.to_dicts()
        hxd.calculations.pml.pml_summary.ratio_pml = df_peril_seasonal.select(pl.sum("selection")).item(0, 0) / df_pml_events.height
        # add am best calculation and summarise

        cat_output_fields = [
            # "index",
            "location_id",
            "country",
            "area",
            "city",
            "year",
            "day",
            "event_id",
            "model_code",
            "peril_set_code",
            "unadj_damage_ratio",
            "description",
            "base_zone_peril",
            "base_zone_peril_factor",
            "event_factor",
            "damage_ratio",
            "month",
            "peril",
            "day_only_loss",
            "indoor_dr",
            "outdoor_dr",
            "indoor_day_loss_pct",
            "outdoor_day_loss_pct",
            "lead_time",
            "indoor_prior_days",
            "outdoor_prior_days",
            "days_before_event",
            "hit_count",
            "affected_tiv",
            "cat_loss_window",
            "pml_contributor",
            "cat_dr_pass",
            # "cat_partial_loss",
            "event_loss_before_recovery",
            "event_loss_after_recovery",
            "event_loss_after_aggs",
            "annual_loss_before_recovery",
            "annual_loss_after_recovery",
            "annual_loss_after_aggs"
        ]

        nat_cat_sims = get_parameters("calculations/cat/sim_count.json")[0]

        # summarise cat losses by event_id and location_id and write to hxd
        hxd.calculations.cat = df_cat_events.select(cat_output_fields).to_dicts()

        df_pivot = pivot_cat_losses(df_index_summary.select(["index", "peril", "loss_portions"]))
        df_pivot = df_pivot.with_columns(
            [pl.col(col).truediv(nat_cat_sims).alias(col) for col in df_pivot.columns if col != "index"]
        )
        metric = "layer_loss"
        expected_cols = [f"{peril}_{metric}" for peril in nat_cat_perils]
        existing_cols = set(df_pivot.columns)
        missing_cols = [col for col in expected_cols if col not in existing_cols]
        if missing_cols:
            df_pivot = df_pivot.with_columns([pl.lit(0.0).alias(col) for col in missing_cols])



        hxd.calculations.cat_event_level_summary_output = df_pivot.sort("index").to_dicts()

        # create cat graph to be shown in results from full table
        
        # summarise pml and pricing contribution for results
    else:
        hx.errors.validation("No Cat events returned for selected locations.")
    return

@hx.task 
def adverse_weather(hxd, progress):
    
    adverse_weather_rating(hxd, progress)
    return

# @hx.task 
def adverse_weather_rating(hxd, progress):
    """Call BEV weather API and join results to event info by thresholds."""
    
    api_key = hx.secrets.bev_api_key_new

    perils = get_parameters("calculations/adverse_weather/modelled_perils.json")

    # extract event set from hxd object
    df_event_info = pd_df_from_hx_list(hxd.event_level.event_info)

    df_params_aw_thresholds = hx.params.adverse_weather_thresholds[
            ["event_type", 
            "event_subset", 
            "venue_type", 
            "region",
            "adverse_weather_severity", 
            "rain_threshold",
            "max_wind_threshold", 
            "max_gust_threshold", 
            "lightning_threshold", 
            "requires_cumulative", 
            "cumulative_rain_days", 
            "cumulative_rain_threshold"]
        ]

    fallback_thresholds = get_parameters("calculations/adverse_weather/fallback_thresholds.json")

    for threshold in list(fallback_thresholds.keys()):
        df_params_aw_thresholds.loc[df_params_aw_thresholds[threshold] == "fallback", threshold] = fallback_thresholds[threshold]

    df_params_aw_thresholds_region = df_params_aw_thresholds[df_params_aw_thresholds["region"] != ""].set_index(["event_type", "event_subset", "venue_type", "region"])
    df_params_aw_thresholds_no_region = df_params_aw_thresholds.loc[
        df_params_aw_thresholds["region"] == "",
        [col for col in df_params_aw_thresholds.columns if col != "region"]].set_index(["event_type", "event_subset", "venue_type"])

    df_event_info_region = df_event_info.set_index(
        ["risk_factors/event_type", "risk_factors/event_subset", "risk_factors/venue_type", "location/area"]
    )
    
    df_event_info_region.update(
        df_params_aw_thresholds_region
    )

    df_event_info_region = df_event_info_region.reset_index()

    df_event_info_no_region = df_event_info.set_index(
        ["risk_factors/event_type", "risk_factors/event_subset", "risk_factors/venue_type"]
    )
    df_event_info_no_region.update(
        df_params_aw_thresholds_no_region
    )

    df_event_info_no_region = df_event_info_no_region.reset_index()

    # replace null values 
    df_event_info = df_event_info_region.combine_first(df_event_info_no_region)
    # Join lat/lon from the new location mapping table
    df_location_mapping = hx.params.area_city_location_id_mapping_new
    
    df_event_info = df_event_info.merge(
        df_location_mapping[["country", "area", "city", "latitude", "longitude"]],
        left_on=["location/country", "location/area", "location/city"],
        right_on=["country", "area", "city"],
        how="left",
    )

    df_event_info['bev_location'] = (
        df_event_info['location/country'] + ", "
        + df_event_info['location/area'] + ", "
        + df_event_info['location/city']
    )

    df_event_set = (
        df_event_info[["index", "bev_location", "event_start_date", "event_end_date",
                        "requires_cumulative", "latitude", "longitude"]]
        .assign(**{
            col: df_event_info[col].dt.strftime('%Y-%m-%d')
            for col in df_event_info.select_dtypes(include=['datetime', 'datetimetz']).columns
        })
        .rename(columns = {
            "bev_location":"location",
            "event_start_date": "start_date",
            "event_end_date": "end_date"})
    )

    # Build daily event payload with lat/lon; location field is empty string for prod API
    daily_event_set = (
        df_event_set[["index", "start_date", "end_date", "latitude", "longitude"]]
        .assign(location="")
        .to_dict(orient="records")
    )

    # Labels for warning messages
    daily_labels = [
        f"{row['location']} ({row.get('latitude', '')}, {row.get('longitude', '')})"
        for row in daily_event_set
    ]

    base_url = _DEFAULT_BASE_URL

    daily_api_json, daily_failed = bev_task_batches_threaded(
        api_key=api_key,
        perils=perils["daily"],
        endpoint="daily",
        event_set=daily_event_set,
        concurrency=_CD_CONCURRENCY,
        batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
        max_combinations=ENDPOINT_MAX_COMBINATIONS["daily"],
        base_url=base_url,
    )

    all_warnings = []
    all_warnings.extend(build_failed_warnings(daily_failed, daily_labels, "daily"))

    # Normalise daily results (probability / 100) into schema shape
    df_daily = pd.DataFrame(normalise_daily_results(daily_api_json)).sort_values('index')

    # write df_daily to hxd
    hxd.calculations.adverse_weather.adverse_weather_daily = df_daily.to_dict(orient="records")

    # Build expanding event payload with lat/lon
    expanding_event_set = (
        df_event_set.loc[
            df_event_set["requires_cumulative"] == 1,
            ["index", "start_date", "end_date", "latitude", "longitude"]
        ]
        .assign(location="")
        .to_dict(orient="records")
    )

    if len(expanding_event_set) > 0:

        expanding_window = get_parameters("calculations/adverse_weather/rolling_window.json")[0]

        expanding_labels = [
            f"({row.get('latitude', '')}, {row.get('longitude', '')})"
            for row in expanding_event_set
        ]

        expanding_api_json, expanding_failed = bev_task_batches_threaded(
            api_key=api_key,
            perils=perils["expanding"],
            window_days=expanding_window,
            endpoint="expanding",
            event_set=expanding_event_set,
            concurrency=_CD_CONCURRENCY,
            batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["expanding"],
            base_url=base_url,
        )

        all_warnings.extend(build_failed_warnings(expanding_failed, expanding_labels, "expanding"))

        # Normalise expanding results (probability / 100) into schema shape
        df_expanding = pd.DataFrame(normalise_expanding_results(expanding_api_json))

        # write df_expanding to hxd
        hxd.calculations.adverse_weather.adverse_weather_expanding = df_expanding.to_dict(orient="records")

    # Surface any failed-event warnings via the framework validation pattern
    if all_warnings:
        df_params = hxd.params.validation.copy()
        df_warnings = df_params[df_params["type"] == "warning"]
        df_warnings = pd.concat(
            [df_warnings, pd.DataFrame(all_warnings)],
            ignore_index=True,
        )
        hxd.outputs.warnings = df_warnings[["label", "message"]].to_dict(orient="records")

    return