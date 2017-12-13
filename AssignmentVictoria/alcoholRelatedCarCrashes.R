
setwd("~/DataScienceAssignments2017/AssignmentVictoria/")

#library(gdata)
library(xlsx)
library(MASS)
#install.packages("rJava", dependencies = TRUE)
#install.packages("xlsx", dependencies = TRUE)
###################################################################
# 1. Identify demographic characteristics of the drivers that are #
# risk (or protective) factors of car accidents.                  #
###################################################################

# Load the data into a variable.
data_car_accidents <- read.xlsx("Data_Car_accidents.xlsx", 1)
data <- data_car_accidents
data$Gender <- as.numeric(data_car_accidents$Gender)
data$Accident[data_car_accidents$Accident == "YES"] <- 1
data$Accident[data_car_accidents$Accident == "NO"] <- 0
data$Socioeconomic_status <- as.numeric(data_car_accidents$Socioeconomic_status)
View(data)
# To identify the demographic characteristics of the drivers that are
# risk factors or protective for car accidents we can use glm for this.
# We do this for all combinations:
#                                 - Age and Accident
#                                 - Gender and Accident
#                                 - SocioeconmicStatus and Accident

# Age and Accident
accident_age <- glm(formula = Accident ~ Age, family = "binomial", data = data)
summary(accident_age)

# The logarithmic coefficient of age is 0.04298 with the p-value < 0.05
age_coefficient <- coefficients(accident_age)[2]
# The logarithmic confidence interval is [0.025, 0.062]
age_ci <- confint(accident_age, parm = c("Age"), level=0.95)

# The natural coefficient (1.044) and confidence interval [1.026, 1.064]
exp(age_coefficient)
exp(age_ci)

# Gender and Accident
accident_gender <- glm(formula = Accident ~ Gender, family = "binomial", data = data)
summary(accident_gender)

# The logarithmic coefficient of gender is 1.5604 with the p-value < 0.05
gender_coefficient <- coefficients(accident_gender)[2]
gender_ci <- confint.default(accident_gender, parm = c("Gender"), level = 0.95)

# The natural coefficient (4.761) and confidence interval [2.277, 9.952]
exp(gender_coefficient)
exp(gender_ci)

# Socioeconomic status and accident
accident_socio <- glm(formula = Accident ~ Socioeconomic_status, family = "binomial", data = data)
summary(accident_socio)

# The logarithmic coefficient of gender is 0.0982 with the p-value > 0.05
socio_coefficient <- coefficients(accident_socio)[2]
socio_ci <- confint(accident_socio, parm = c("Socioeconomic_status"), level = 0.95)

# The natural coefficient (1.103) and confidence interval [0.772, 1.582]
exp(socio_coefficient)
exp(socio_ci)

##########################################################################
# 2. Obtain the model relating BAC and car accidents (both, not adjusted #
# and adjusted for confounders). Interpret the not‐adjusted and adjusted #
# odds ratios. Is there a significant association between BAC and car    # 
# accidents?                                                             #
##########################################################################

# We find confounders for the adjusted model by checking if they fulfill the three confounder conditions
# - Associated with BAC
# - Risk factor for Accident (independent of BAC)
# - Not intermediate factor (BAC is not depended on potential confounder)
# The potential confounder: Gender, Age, Socioeconomin status

# Condition 1:
# Find association between each potential confounder and BAC by doing a general logistic model

# Age and BAC
bac_age <- glm(formula = BAC ~ Age, data = data)
summary(bac_age)

# Gender and BAC
bac_gender <- glm(formula = BAC ~ Gender, data = data)
summary(bac_gender)

# There is a significant association between age and BAC, and between gender and BAC. 
# This fulfills the first condition

# Condition 2:
# Both age and gender were shown in the previous exercise to be significantly associated with Accidents 
# independently from BAC. This means they both are risk factors for Accident and fulfills the second condition.

# Condition 3:
# Age and Gender are not intermediate factors, because intuitively blood-alcohol content cannot affect
# the age or gender

# All the confounder condition are by age and gender and can therefore be considered confounders

# Unadjusted Model
accident_bac <- glm(formula = Accident ~ BAC, family = "binomial", data = data)
summary(accident_bac)
# The logarithmic coefficient of BAC is 4.00 with the p-value < 0.05
bac_coefficient <- coefficients(accident_bac)[2]
# The logarithmic confidence interval is [3.03, 5.14]
bac_ci <- confint(accident_bac, parm = c("BAC"), level=0.95)

# The natural coefficient (54.70) and confidence interval [20.77, 171.21]
exp(bac_coefficient)
exp(bac_ci)

# Adjusted Model
accident_bac_adjusted <- glm(formula = Accident ~ BAC + Gender + Age, family = "binomial", data = data_car_accidents)
summary(accident_bac_adjusted)

# The logarithmic coefficient of BAC is 3.78, Gender is 0.92 and Age is 0.022 with the p-value < 0.05
adjusted_coefficient <- coefficients(accident_bac_adjusted)
# The logarithmic confidence interval of BAC is [2.77, 4.97], Gender is [-0.054, 1.93] and Age is [-0.0018, 0.047]
adjusted_ci <- confint(accident_bac_adjusted, parm = c("BAC", "Gender", "Age"), level=0.95)

# The natural coefficient of BAC is 43.76, Gender is 2.51 and Age is 1.02 
# and confidence interval for BAC is [15.90, 144.08], Gender is [0.95, 6.91] and Age is [1.00, 1.05]
exp(adjusted_coefficient)
exp(adjusted_ci)

# Conclusion: 
# Unadjusted model shows the BAC odds ratio is 54.7
# When BAC is increased by 1 the subject has 54.7 times higher odds of getting into an accident.
# Adjusted model shows the BAC odds ratio is 43.7
# When BAC is increased by 1 the subject has 43.7 times higher odds of getting into an accident.

########################################################################
# 3. Is there any other potential confounder (not included in the file #
# ”Data_car_accidents”) that should have been considered in the study? #
# How would you include it in the analysis?                            #
########################################################################

# Weight (scale)
# Poor Vision (Yes/No)
# Medication not appropriate for driving (Yes/No)
# Physical condition (Good/Bad)

######################################################################
# 4. Plot the unadjusted (crude) and adjusted models. Comment on the #
# similarities or differences between the two.                       #
######################################################################



##################################################################
# 5. What is the probability that a 40 yr male whose BAC is >1‰, #
# causes a car accident? What will be the probability, 10, 20,   #
# 30 and 40 years later? Is this change linear?                  #
##################################################################


###################################################################
# 6. We obtain information on a new set of drivers (17 subjects). #
# Evaluate the predictive performance of the model by calculating #
# the accuracy, sensitivity, specificity and precision of the     #
# model using this new dataset. Consider the threshold value for  #
# the probability as equal to 0.5. The data for the 17 subjects   #
# is in the Excel file ”Data_car_accidents” (sheet: ”Data_17_     #
# subjects”).                                                     #
###################################################################