Understanding Export Pricing:

This file contains 20-years of hourly export rates for three sets of PG&E Solar Billing Plan vintages (i.e. Net Billing Tariff / NBT), NBT23, NBT24, and NBT00. 
-	NBT23 = Solar Bill Plan Customers with applications filed in 2023 that qualify for 9-year lock-in export rates
-	NBT24 = Solar Billing Plan Customers with applications filed in 2024 that qualify for 9-year lock-in export rates.
-	NBT00 = Solar Billing Plan Customers that do not qualify for any 9-year lock-in export rates.
The dates and times associated with the export rates are presented in UTC time, covering from 1/1/2024 12:00:00am PST, to 12/31/2043 11:59:59pm PST.
Per CPUC Resolution E-5301, PG&E is required to provide 20-years of export rates. However please note that NBT23 and NBT24 vintage customers are only guaranteed export rates for 9-years from the Permission-To-Operate (PTO) date of your system. Any rate factors in this file beyond the  9-year lock-in period are for illustrative purposes only and are not actual effective SBP Export Rates at those times. For NBT00 customers, only those rates for Pacific Standard Time calendar year 2024 are actual effective rates, and all rates after that are for illustrative purposes only.

Files Types:

Per CPUC Resolution E-5301, this file contains the export rates in both .csv and .xml formats.


Data Fields Description: 

The following is a description of the data fields in the data file.

RateLookupID Column:

	If the RateLookupID includes “USCA-PGXX” that indicates Delivery Export Rates.

	If the RateLookupID includes “USCA-XXPG” that indicate Generation Export Rates*.

	*Generation Export Rates are only applicable to SBP customers that receive bundled generation service from PG&E. Customers that receive generation service from a Community Choice Aggregator (CCA) or a Direct Access (DA) provider should refer to that generation service provider for more information about the generation export pricing available to them.


RateName Column: 
	The number that follows “NBT” represents the legacy pricing for that year. Please see first paragraph for a description.

Dates and Times Columns:
	DateStart, TimeStart, DateEnd and TimeEnd values are in Coordinated Universal Time (UTC). 

DayStart, DayEnd and ValueName are in Pacific Prevailing Time:
	-	These fields indicate the effective day-type categories of the rate factor
	-	Monday through Sunday are represented as 1-7. Holidays are listed as number “8” in the DayStart and DayEnd columns. 
	-	ValueName Column indicates the month and weekday hour or weekend hour starting value for the rate factor.


Value and Unit Columns:
	This represents the dollar amount pricing per export kWh.
