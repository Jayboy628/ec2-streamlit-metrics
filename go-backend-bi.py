import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import DataFrame
from awsglue.job import Job
from pyspark.sql.window import Window
import os
import json
import boto3
from datetime import datetime
from urllib.parse import urlparse
from pyspark.sql.functions import col, lower, trim, udf, lit, when, coalesce, datediff, avg, max, min, count, max as spark_max,  sum as spark_sum, datediff, first, countDistinct, row_number
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql import SparkSession


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    load_aws_credentials_from_secrets(AWS_SECRET_NAME, AWS_PROFILE, AWS_REGION)
    spark = create_spark_session()
    print_s3_config(spark)

    # Replace with your actual S3 path
    parquet_path = "s3a://gp-elt-657082399901-dev/final/"
    df_order_items = load_order_data(spark, parquet_path)


## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ---------------------------
# 1. Rearrange Columns
# ---------------------------


eexpected_columns_final = [
    "order_key", "date_key", "app_name", "restaurant_id", "user_id", "printed_card_number", "order_id", "lineitem_id",
    "is_loyalty", "item_category", "item_name", "size", "option_group_name","option_name", "option_price", "option_quantity",  "item_price", "item_quantity", "time", "date",
    "month_name","day_of_weeks", "is_weekends", "is_holidays", "holiday_name"
]


def rearrange_columns(df: DataFrame, expected_columns: list) -> DataFrame:
    """
    Aligns DataFrame to match expected column order:
    - Adds missing columns as nulls
    - Reorders columns
    - Ensures no duplicate column names

    Args:
        df (DataFrame): Input Spark DataFrame
        expected_columns (list): List of expected column names and order

    Returns:
        DataFrame: Reordered and aligned DataFrame
    """
    current_columns = set(df.columns)
    # Add only truly missing columns
    for col_name in expected_columns:
        if col_name not in current_columns:
            df = df.withColumn(col_name, lit(None))
    
    # Select columns in exact order, only if they exist
    final_columns = [col for col in expected_columns if col in df.columns]
    df = df.select(*final_columns)

    print("Final columns set to:")
    print(df.columns)
    return df


# ---------------------------
# 2. Customer Lifetime Value
# ---------------------------

# Step 1: Compute revenue per order
new_spark_order_items = new_spark_order_items.withColumn(
    "revenue_per_order",
    (coalesce(col("option_price"), lit(0)) * coalesce(col("option_quantity"), lit(0))) +
    (coalesce(col("item_price"), lit(0)) * coalesce(col("item_quantity"), lit(1)))
)

# Step 2: Filter out guest users
filtered_orders = new_spark_order_items.filter(col("customer_id") != "_guest")

# Step 3: Group and aggregate
revenue_per_order = filtered_orders.groupBy("restaurant_id","customer_id", "date_key", "is_loyalty").agg(
    sum("revenue_per_order").alias("total_revenue"),
    count("order_id").alias("total_orders"),
    max("date").alias("max_date"),
    avg("revenue_per_order").alias("avg_revenue_per_order"),
    min("revenue_per_order").alias("min_revenue_per_order")
)

# write the final DataFrame to S3
output_path = "s3a://gp-elt-657082399901-dev/metrics/revenue_per_order/"
revenue_per_order.write.mode("overwrite").parquet(output_path)


# ---------------------------
# 3. CLV: GROUP VALUES
# High CLV: Top 20% customers
# Medium CLV: Mid 60%
# Low CLV: Bottom 20%
# ---------------------------


def tag_clv_by_restaurant_spark(df: DataFrame, clv_col: str = "total_revenue") -> DataFrame:
    """
    Tags customers with CLV buckets (Low/Medium/High) per restaurant_id using quantile thresholds.
    
    Args:
        df (DataFrame): Input Spark DataFrame with columns `restaurant_id` and `clv_col`.
        clv_col (str): Column used to compute quantiles for CLV tagging.

    Returns:
        DataFrame: Original DataFrame with a new column `clv_tag`.
    """
    restaurant_ids = [row["restaurant_id"] for row in df.select("restaurant_id").distinct().collect()]
    threshold_data = []

    for rid in restaurant_ids:
        sub_df = df.filter(col("restaurant_id") == rid)
        q20, q80 = sub_df.approxQuantile(clv_col, [0.2, 0.8], 0.01)
        threshold_data.append((rid, q20, q80))

    # Create a DataFrame of thresholds
    threshold_df = df.sparkSession.createDataFrame(threshold_data, ["restaurant_id", "low_threshold", "high_threshold"])

    # Join original data with thresholds
    df_joined = df.join(threshold_df, on="restaurant_id", how="left")

    # Apply tagging
    df_tagged = df_joined.withColumn(
        "clv_tag",
        when(col(clv_col) >= col("high_threshold"), lit("High CLV"))
        .when(col(clv_col) <= col("low_threshold"), lit("Low CLV"))
        .otherwise(lit("Medium CLV"))
    )

    # Optional: drop intermediate threshold columns
    df_tagged = df_tagged.drop("low_threshold", "high_threshold")

    return df_tagged

# write the final DataFrame to S3
output_path = "s3a://gp-elt-657082399901-dev/metrics/clv_tagged/"
tagged_df.write.mode("overwrite").parquet(output_path)

# ----------------------------------------------------------------------------
# 4. Customer Segmentation & Behavior: Use RFM logic based on order_items:
#
## Recency: Days since last purchase
## Frequency: Number of purchases in last N months
## Monetary: Total spend in last N months
#
# ----------------------------------------------------------------------------


from pyspark.sql.functions import (
    col, max as spark_max, count, sum as spark_sum, datediff, lit, first
)
from pyspark.sql import DataFrame

def rfm_analysis_spark(
    df: DataFrame,
    customer_col: str = "customer_id",
    restaurant_col: str = "restaurant_id",
    date_col: str = "date",
    revenue_col: str = "revenue_per_order",
    loyalty_col: str = "is_loyalty"
) -> DataFrame:
    """
    Perform RFM analysis grouped by restaurant_id and customer_id using PySpark.
    Includes loyalty flag if consistent per group.
    """
    df = df.filter(col(customer_col) != "_guest")

    # Snapshot date for recency
    snapshot_date = df.agg(spark_max(col(date_col)).alias("snapshot")).collect()[0]["snapshot"]
    snapshot_date_lit = lit(snapshot_date)

    # RFM Aggregation
    rfm_df = df.groupBy(restaurant_col, customer_col).agg(
        datediff(snapshot_date_lit, spark_max(col(date_col))).alias("recency"),
        count("*").alias("frequency"),
        spark_sum(col(revenue_col)).alias("monetary"),
        first(col(loyalty_col)).alias("is_loyalty")  # Take first known loyalty status
    )

    return rfm_df



# -------------------------------------------------
# 5. Customer Segmentation & Behavior: Segment:
## VIPs: High R, F, M
## New Customers: Low F, high R
## Churn Risk: Low R, low F
#
# --------------------------------------------------

def segment_customers_spark(df: DataFrame) -> DataFrame:
    """
    Segment customers into groups based on RFM metrics.
    """
    return df.withColumn(
        "segment",
        when((col("recency") <= 180) & (col("frequency") > 50) & (col("monetary") > 500), "VIP")
        .when((col("recency") <= 180) & (col("frequency") <= 50) & (col("monetary") > 200), "New Customer")
        .when((col("recency") > 180) & (col("frequency") <= 5) & (col("monetary") <= 1000), "Churn Risk")
        .otherwise("Other")
    )
rfm_df = rfm_analysis_spark(new_spark_order_items)
segmented_df = segment_customers_spark(rfm_df)

# write the materics to S3
output_path = "s3a://gp-elt-657082399901-dev/metrics/rfm_segmented/"
segmented_df.write.mode("overwrite").parquet(output_path)

# ---------------------------
# 6. customer activity profile grouped by rest and cust: For each customer_id, compute
## Days since last order
## Average gap between orders
## % change in spend over last N periods
## Tag customers based on inactivity thresholds (e.g., >45 days = “at risk”)
#
# ---------------------------

def customer_activity_profile_spark(
    df: DataFrame,
    customer_col: str = "customer_id",
    restaurant_col: str = "restaurant_id",
    date_col: str = "date",
    revenue_col: str = "revenue_per_order"
) -> DataFrame:
    """
    Build customer activity profile grouped by restaurant and customer using PySpark.
    """

    # Filter out guest users
    df = df.filter(col(customer_col) != "_guest")

    # Define window partitioned by restaurant and customer, ordered by date
    window_spec = Window.partitionBy(restaurant_col, customer_col).orderBy(col(date_col))

    # Calculate time difference between orders (gap in days)
    df = df.withColumn("order_gap", datediff(col(date_col), lag(col(date_col)).over(window_spec)))

    # Calculate revenue change (%)
    df = df.withColumn("prev_revenue", lag(col(revenue_col)).over(window_spec))
    df = df.withColumn(
        "revenue_change_pct",
        when(col("prev_revenue").isNotNull(), (col(revenue_col) - col("prev_revenue")) / col("prev_revenue"))
    )

    # Step 1: Last order date
    last_order_df = df.groupBy(restaurant_col, customer_col).agg(
        spark_max(col(date_col)).alias("last_order_date")
    )

    # Step 2: Average gap between orders
    avg_gap_df = df.groupBy(restaurant_col, customer_col).agg(
        avg("order_gap").alias("avg_gap_days")
    )

    # Step 3: Average % revenue change
    avg_rev_change_df = df.groupBy(restaurant_col, customer_col).agg(
        avg("revenue_change_pct").alias("avg_pct_change")
    )

    # Step 4: Join all metrics
    profile = last_order_df \
        .join(avg_gap_df, on=[restaurant_col, customer_col], how="left") \
        .join(avg_rev_change_df, on=[restaurant_col, customer_col], how="left")

    # Step 5: Days since last order
     # Get the max date from the dataset (latest date in history)
    max_date_df = df.select(spark_max(col(date_col)).alias("max_dataset_date"))
    max_dataset_date = max_date_df.collect()[0]["max_dataset_date"]

    # Broadcast the date as a literal and compute days since last order
    profile = profile.withColumn(
        "days_since_last_order",
        datediff(lit(max_dataset_date), col("last_order_date"))
    )

    # Step 6: Activity tagging
    risk_threshold = 45
    profile = profile.withColumn(
        "recency_tag",
        when(col("days_since_last_order") > risk_threshold, "inactive").otherwise("active")
    ).withColumn(
        "gap_tag",
        when(col("avg_gap_days") > risk_threshold, "at risk").otherwise("healthy")
    ).withColumn(
        "activity_tag",
        when((col("recency_tag") == "inactive") | (col("gap_tag") == "at risk"), "at risk")
        .otherwise("active")
    )

    return profile

activity_profile_df = customer_activity_profile_spark(new_spark_order_items)
# activity_profile_df.show(2, truncate=False, vertical=True)

# write to S3
output_path = "s3a://gp-elt-657082399901-dev/metrics/customer_profile/"
activity_profile_df.write.mode("overwrite").partitionBy("restaurant_id").parquet(output_path)



# --------------------------------------------------------------------------
# 7. Sales Trends Monitoring:
## Aggregate daily, weekly, and monthly revenue from order_items:
## Location
## Menu category (if available)
## Time of day (optional)
# --------------------------------------------------------------------------
def sales_trends_monitoring_spark(df, datetime_col="date", s3_output_path=None):
    from pyspark.sql.functions import (
        col, to_date, to_timestamp, hour, weekofyear, year, month, date_format,
        sum as spark_sum, coalesce, lit, concat_ws, expr
    )
    from pyspark.sql.functions import lpad

    # --- 1. Compute revenue and parse time ---
    df = df.withColumn("date", to_date(col("date"))) \
        .withColumn("revenue", 
            coalesce(col("item_price"), lit(0.0)) * coalesce(col("item_quantity"), lit(0)) +
            coalesce(col("option_price"), lit(0.0)) * coalesce(col("option_quantity"), lit(0))
        )

    # --- 2. Extract date parts ---
    df = df.withColumn("year", year("date")) \
        .withColumn("month", date_format("date", "MMMM")) \
        .withColumn("week", weekofyear("date")) \
        .withColumn("day", col("day_of_weeks")) \
        .withColumn("hour", hour(to_timestamp("time", "HH:mm:ss")))

    # --- 3. Aggregations by granularity ---

    # Daily
    daily_df = df.groupBy("year", "day", "date", "restaurant_id", "item_category") \
        .agg(spark_sum("revenue").alias("revenue")) \
        .withColumn("granularity", lit("daily"))

    # Weekly (approximate start date of the week)
    weekly_df = df.groupBy("year", "week", "restaurant_id", "item_category") \
        .agg(spark_sum("revenue").alias("revenue")) \
        .withColumn("date", expr("date_add(to_date(concat(year, '-01-01')), (week - 1) * 7)")) \
        .withColumn("granularity", lit("weekly"))

    # Monthly
    monthly_df = df.groupBy("year", "month", "restaurant_id", "item_category") \
        .agg(spark_sum("revenue").alias("revenue")) \
        .withColumn("month_num", month(to_date(concat_ws(" ", col("month"), col("year")), "MMMM yyyy"))) \
        .withColumn("date", to_date(concat_ws("-", col("year"), col("month_num"), lit("01")))) \
        .withColumn("granularity", lit("monthly"))

    # Hourly
    hourly_df = df.groupBy("year", "hour", "day", "date", "restaurant_id", "item_category") \
        .agg(spark_sum("revenue").alias("revenue")) \
        .withColumn("granularity", lit("hourly"))

    # --- 4. Write each to S3 with restaurant_id partition --- 
    if s3_output_path:
        s3_output_path = s3_output_path.rstrip("/") + "/"
        if not s3_output_path.startswith("s3a://"):
            raise ValueError("s3_output_path must start with 's3a://'")


        daily_df.repartition("restaurant_id").write.mode("overwrite") \
            .partitionBy("restaurant_id").parquet(f"{s3_output_path}daily/")
        print("Daily revenue written.")

        weekly_df.repartition("restaurant_id").write.mode("overwrite") \
            .partitionBy("restaurant_id").parquet(f"{s3_output_path}weekly/")
        print("Weekly revenue written.")

        monthly_df.repartition("restaurant_id").write.mode("overwrite") \
            .partitionBy("restaurant_id").parquet(f"{s3_output_path}monthly/")
        print("Monthly revenue written.")

        hourly_df.repartition("restaurant_id").write.mode("overwrite") \
            .partitionBy("restaurant_id").parquet(f"{s3_output_path}hourly/")
        print("Hourly revenue written.")

    return daily_df, weekly_df, monthly_df, hourly_df

s3_output_path = "s3a://gp-elt-657082399901-dev/metrics/sales_trends/"
daily_df, weekly_df, monthly_df, hourly_df = sales_trends_monitoring_spark(
    new_spark_order_items,
    datetime_col="date",
    s3_output_path=s3_output_path
)




# ---------------------------
# 8. Loyalty Program Impact: Compare loyalty members vs non-members in terms of spend and engagement.
## Filter order_items by is_loyalty = true vs false
## Compare per-customer:
### Average Spend
### Repeat Orders
### Lifetime Value
# ---------------------------
# Filter out guest users
filtered_df = new_spark_order_items.filter(col("customer_id") != "_guest")

# First: group by restaurant + loyalty + customer
impact_df = filtered_df.groupBy("restaurant_id", "is_loyalty", "customer_id").agg(
    avg(col("revenue_per_order")).alias("average_spend"),
    count("customer_id").alias("repeat_orders"),
    spark_sum("revenue_per_order").alias("lifetime_value")
)

# write to metrics
output_path = "s3a://gp-elt-657082399901-dev/metrics/loyalty_program_impact/"
impact_df.write.mode("overwrite").parquet(output_path)

# Second: summarize by restaurant + loyalty group
impact_summary = impact_df.groupBy("restaurant_id", "is_loyalty").agg(
    avg("average_spend").alias("avg_spend_per_customer"),
    avg("repeat_orders").alias("avg_repeat_orders"),
    avg("lifetime_value").alias("avg_lifetime_value")
)

# write to metrics
output_summary_path = "s3a://gp-elt-657082399901-dev/metrics/loyalty_program_impact_summary/"
impact_summary.write.mode("overwrite").parquet(output_summary_path)


# ---------------------------
# 9. Top-Performing Locations: Identify best and worst-performing store locations.
## Group order_items by location_id (or store_id if available)
## Total revenue
## Average order value
## Orders per day/week
## Rank locations based on revenue
# ---------------------------

from pyspark.sql.functions import col, sum as spark_sum, avg, countDistinct, row_number
from pyspark.sql.window import Window

def top_performing_locations_spark(df, location_col='restaurant_id', date_col='date', revenue_col='revenue_per_order'):
    from pyspark.sql.window import Window
    from pyspark.sql.functions import col, sum as spark_sum, avg, countDistinct, row_number

    location_metrics = df.groupBy(location_col).agg(
        spark_sum(col(revenue_col)).alias("total_revenue"),
        avg(col(revenue_col)).alias("average_order_value"),
        countDistinct(col(date_col)).alias("active_days"),
        countDistinct("order_id").alias("total_orders")
    )

    location_metrics = location_metrics.withColumn(
        "orders_per_day",
        col("total_orders") / col("active_days")
    )

    window_spec = Window.orderBy(col("total_revenue").desc())
    ranked = location_metrics.withColumn("rank", row_number().over(window_spec))

    return ranked.select(
        location_col, "total_revenue", "average_order_value", "orders_per_day", "rank"
    )

top_locations = top_performing_locations_spark(new_spark_order_items)
# write to metrics top locations
output_top_locations_path = "s3a://gp-elt-657082399901-dev/metrics/top_locations/"
top_locations.write.mode("overwrite").parquet(output_top_locations_path)

# ---------------------------
# 10. Pricing & Discount Effectiveness
## Use order_item_options.option_price to detect discounts (option_price < 0)
## Revenue from discounted orders vs non-discounted
## Number of orders before/after applying discounts
# ---------------------------

def pricing_discount_effectiveness(df, option_price_col="option_price", revenue_col="revenue_per_order", order_col="order_id"):
    from pyspark.sql.functions import col, when, sum as spark_sum, countDistinct, avg

    df = df.withColumn("is_discounted", when(col(option_price_col) < 0, True).otherwise(False))

    summary = df.groupBy("restaurant_id", "is_discounted").agg(
        spark_sum(col(revenue_col)).alias("total_revenue"),
        countDistinct(col(order_col)).alias("total_orders"),
        avg(col(revenue_col)).alias("avg_order_value")
    )

    return summary

discount_summary = pricing_discount_effectiveness(new_spark_order_items)
# write to metrics discount effectiveness
output_discount_path = "s3a://gp-elt-657082399901-dev/metrics/discount_effectiveness/"
discount_summary.write.mode("overwrite").parquet(output_discount_path)






job.commit()