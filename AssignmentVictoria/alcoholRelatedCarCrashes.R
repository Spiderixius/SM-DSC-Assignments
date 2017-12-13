setwd("~/Documents/DataScienceAssignments2017/AssignmentVictoria/")

library(gdata)
#install.packages("rJava", dependencies = TRUE)
#install.packages("xlsx", dependencies = TRUE)
###################################################################
# 1. Identify demographic characteristics of the drivers that are #
# risk (or protective) factors of car accidents.                  #
###################################################################

# Load the data into a variable.
the_data <- read.xls("Data_Car_accidents.xlsx")

# To identify the demographic characteristics of the drivers that are
# risk factors or protective for car accidents we can use glm for this.
# We do this for all combinations:
#                                 - Age and Accident
#                                 - Gender and Accident
#                                 - SocioeconmicStatus and Accident
#                                 - BAC and Accident

# Age and Accident
accident_age <- glm(formula = Accident ~ Age, family = binomial(logit), data = the_data)
summary(accident_age)
exp(coefficients(accident_age))
exp(confint.default(accident_age))

# Gender and Accident
accident_gender <- glm(formula = Accident ~ Gender, family = binomial(logit), data = the_data)
summary(accident_gender)
exp(coefficients(accident_gender))
exp(confint.default(accident_gender))

# Socioeconomic status and accident
accident_soceco_status <- glm(formula = Accident ~ Socioeconomic_status, family = binomial(logit), data = the_data)
summary(accident_soceco_status)
exp(coefficients(accident_soceco_status))
exp(confint.default(accident_soceco_status))

# BAC and accident
accident_bac <- glm(formula = Accident ~ BAC_., family = binomial(logit), data = the_data)
summary(accident_bac)
exp(coefficients(accident_bac))
exp(confint.default(accident_bac))

##########################################################################
# 2. Obtain the model relating BAC and car accidents (both, not adjusted #
# and adjusted for confounders). Interpret the not‐adjusted and adjusted #
# odds ratios. Is there a significant association between BAC and car    # 
# accidents?                                                             #
##########################################################################

# From previous exercise we can see that Age and Gender has a statistical
# significance for whether there is an accident. Socioeconomic does not.
# Therefore we will do an association test on Age and Gender in relation
# to BAC levels.

t.test(BAC_. ~ Gender, alternative="two.sided", conf.level=.95, var.equal=FALSE, data = the_data)

# Age and BAC
bac_age <- glm(formula = BAC_. ~ Age, data = the_data)
summary(bac_age)

# Gender and BAC
bac_gender <- glm(formula = BAC_. ~ Gender, data = the_data)
summary(bac_gender)

# Unadjusted
# BAC and accident
accident_bac <- glm(formula = Accident ~ BAC_., family = binomial(logit), data = the_data)
summary(accident_bac)
exp(coefficients(accident_bac))
exp(confint.default(accident_bac))

# Adjusted
accident_bac_adjusted <- glm(formula = Accident ~ BAC_. + Gender + Age, family = binomial(logit), data = the_data)
summary(accident_bac_adjusted)
exp(coefficients(accident_bac_adjusted))
exp(confint.default(accident_bac_adjusted))

########################################################################
# 3. Is there any other potential confounder (not included in the file #
# ”Data_car_accidents”) that should have been considered in the study? #
# How would you include it in the analysis?                            #
########################################################################


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