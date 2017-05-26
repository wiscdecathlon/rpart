#This app is an original shiny app called "Elective Cath Risk Predictor" 
#It employs machine learning techniques: 'rpart' package,  and intended to be  
#hosted for educational use on a shiny server--for instance, shinyapps.io 

# It is created for educational purposes during graduate degree program, 
# and in current form, is not intended for direct patient care.
# Copyright (C) 2017 Nathaniel P. Brown MD
# Project contributors include: Christina Mannerheim, Kishore Patel, Martîn Krämér,
#  Thibaud Richard, and Raghed Bitaar

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# load required libraries
library(shiny)
library(rpart)
library(rpart.plot)
library(caret)
library(stats)
library(e1071)
library(plyr)

# Define the UI for application 
 ui <- fluidPage(
       # Custom HTML header
         div(class = "header",
         includeHTML("header.html")
       ),

      # Sidebar calculator entry area
   sidebarLayout(
      sidebarPanel(
        htmlOutput("prediction",container = tags$div,class="well", style="background:#EFEFEF;width:100%;"),
        radioButtons("sex","Gender:",
                     choices = list("Male" = "Male", "Female" = "Female"), 
                     selected = "Male"),
        sliderInput("age", "Age:", min = 20,  max = 110, value = 50),
        selectInput('cp', "Symptoms:", c(Choose='', c("Exertional (cp-1)"="1", "Atypical (cp-2)"="2","Momentary (cp-3)"="3","Absence of Pain"="None")), selectize=TRUE),
        sliderInput("trestbps", "Systolic Blood Pressure:", min = 70,  max = 250, value = 110),
        sliderInput("chol", "Total Cholesterol:", min = 120,  max = 400, value = 180),
        selectInput('fbs', "Blood Sugar:", c(Choose='', c("Normal"="Normal", "Diabetic Range"="High")), selectize=TRUE),
        selectInput('restecg', "Rest EKG Features:", c(Choose='', c("Normal"="Normal", "ST segment depression"="ST segment","Left Ventricular Hypertrophy"="LVH")), selectize=TRUE)
       ),
      #Main display area
      mainPanel(
        
        #1st subtab
        tabsetPanel(
          tabPanel("About", 
                   h3("What Is It?"),
                   p("This app is a risk calculator that uses Machine Learning to determine the likelihood that coronary stenosis will actually be 
                   found in a candidate patient for elective angiography."),
                   
                   h3("How Is It Used?"),
                   p("Enter patient pre-test attributes (de-identified) at left to calculate a risk prediction. Customize the data SOURCE(s) if desired,
                   view the MODEL to better understand prediction decisions, and observe the dependence of model VALIDITY on data sources."),
                   
                   h3("How Does It Work?"),
                   p("Machine Learning refers to techniques of computation used to predict future events 
                  from analysis of past data. In this app, a prediction MODEL is created from pre-test data 
                  SOURCE(s), and is applied to a new patient (at left), to compute an expected probability of having a positive cardiac catheterization.
                     The source data for the model is configurable."),
                   
                   h3("To Whom Does It Apply?"),
                   p("The source data sets were collected from new cardiology patients, with no prior (known) coronary artery disease. Source data is included from
                  4 geographic areas.Each source data set contains patients scheduled for an elective cardiac catheterization, and with NO previous non-invasive 
                  (i.e. stress) test. The model predicts the probability of whether a cath candidate will or will not be found to have coronary disease of >50% stenosis on the cath. 
                  Low-probability results may indicate benefit to choosing a plan of non-invasive testing and medical management instead, coupled with continued primary prevention."),
                   
                   h3("To Whom Does It NOT Apply?"),
                   p("Due to inclusion criteria of the source data, this does not generalize to the general public or lower risk patients who are not thought 
                     to be candidates for angiography. Furthermore, it is not applicable to the setting of acute myocardial infarction.")),
                   
          #2nd subtab        
          tabPanel("Source",
                   h2("Select Source:"),                  
                   # Show a plot of the individual within the training data
                   flowLayout(checkboxGroupInput("dataGroup", label = h3("Data for Model:"), 
                                      choices = list("Cleveland Clinic (US)" = 1, "Hungary" = 2, "VA Long Beach (US)" = 3, "Switzerland"= 4),
                                      selected = c(1)),
                   checkboxGroupInput("painFilter", label = h3("Filter by Symptoms:"), 
                                                                             choices = list("Include patients with pain" = 1, "Include patients without pain" = 2),
                                                                             selected = c(1:2)),
                   htmlOutput("summary",container = tags$div,class="well", style="background:#EFEFEF;width:100%;")),
                                      h3("The Currently Selected Source Patients, Used to Train the Model:"),
                                      plotOutput('plot1')
                                      ),
          #3rd subtab
          tabPanel("Model", 
                   h3("The Model: An \"rpart\" Classification Tree Created from Data Selection:"),
                   plotOutput('plot2'),
                   
                   h3("Candidate patient values:"),
                   tableOutput("values")
                 ),
          #4th subtab
          tabPanel("Validity", 
                   
                   h4("Background on Evaluation of Learning Models:"),
                   
                   p("It is important to evaluate machine learning models to show their effectiveness at predicting new events. Cross-validation is an 
                     internal sampling method for evaluating whether a model is successful making predictions on new data elements.  The statistics are 
                     provided below are 95% confidence intervals based on 500 randomly-seeded repeats of k-fold cross validations, where k=10. 
                     In other words, a random 10% subset of the source selection is tested on a model trained on the other 90%, 5000 times."),
                   
                   p("Poor results in cross validation are caused by imbalance of disease status in the source selection, an absence of reliable predictors,
                     or a low overall number of individuals in data selected. As such, the models are generally weak for females, applying filtering by symptoms, 
                     and using the VA and Switzerland data. Counterintuitively, the absence of pain is very a strong predictor of disease in these data sets. 
                     This may indicate a sampling bias. Thus additional consideration of issues related to clinical validity is also required. In many cases, 
                    the primary factor for the model's predictive capacity is that the presence of pain appears to be a protective factor."),
                   
                   htmlOutput("validation",container = tags$div,class="well", style="background:#EFEFEF;width:80%;"), 
                   
                   htmlOutput("valid", style="width:80%")
                   )
            )
          )
       
      ), theme = "bootstrap.css"
   )
 


# Defines the server function 
server <- function(input, output) {
  
# Reactive expression: to compose a training data subset of  filtered() <- subset(raw) 
  #containing the values of 1) appropriate gender, 
  # 2) selections of data group, and 3) a pain filter
  makeReactiveBinding("raw") 
  filtered <- reactive({
    
    #declare ui inputs
    input$sex
    input$dataGroup
    input$painFilter
    
    isolate({
      #declarative of empty data frame "raw" 
      raw <- data.frame(ID = integer(),
                        age = integer(), 
                        sex = factor(),
                        cp = factor(),
                        trestbps = integer(),
                        chol = integer(),
                        fbs = factor(),
                        restecg = factor(),
                        thalach = integer(),
                        exang = factor(),
                        oldpeak = double(),
                        slope = factor(),
                        ca = factor(),
                        thal = factor(),
                        class = factor(),
                        cv_disease = factor(),
                        color_logic = integer(),
                        line_logic = double(),
                        mean.of.chol.bps = double(),
                        stringsAsFactors=FALSE) 
      
      # conditional expression to compose raw of ui selected, desired csv data
      if (1 %in% input$dataGroup){
      raw <- rbind (raw, read.csv("cleveland_labels.csv",na.strings = "?")) 
      }
      if (2 %in% input$dataGroup){
      raw <- rbind (raw, read.csv("hungarian_labels.csv",na.strings = "?"))
      }
      if (3 %in% input$dataGroup){
        raw <- rbind (raw, read.csv("va.csv",na.strings = "?"))
      }
      if (4 %in% input$dataGroup){
        raw <- rbind (raw, read.csv("switzerland.csv",na.strings = "?"))
      }
      
      # subset by current sex 
      raw <- subset(raw, sex %in% input$sex)
      
      # subset by pain status 
      noPain <- subset(raw, cp %in% c("None"))
      pain <- subset(raw, cp %in% c("1","2","3"))
      
      # conditional for recombining pain subsets if ui is selected 
      if ((1 %in% input$painFilter) && (2 %in% input$painFilter)){
        raw <- rbind (noPain, pain)
      }else if ((1 %in% input$painFilter) && !(2 %in% input$painFilter)){
        raw <- pain
      }else if (!(1 %in% input$painFilter) && (2 %in% input$painFilter)){
        raw <- noPain
      }
      
      })
    }) 
 
#A mini Summary of the class balance "Normal" v ">50% Blockage" for the selected data set. 
output$summary <-renderTable({
  
    frame <-count(filtered(),vars = c("cv_disease"))
   names(frame)<-c("Source Breakdown","# of Patients")
   return(frame)
  
})

  
#A reactive expression to compose of Test Patient object "test"
  test <- reactive({
    
    # Compose data frame from input
    data.frame(
      age = input$age, 
      sex = input$sex,
      cp = input$cp,
      trestbps = input$trestbps,
      chol = input$chol,
      fbs = input$fbs,
      restecg = input$restecg,
      mean.of.chol.bps = (input$chol + input$trestbps)/2,  
      stringsAsFactors=FALSE)
    
  }) 
  
#Reactive expression to compose the main rpart machine learning Model, an rpart object: "rtree" 
  rtree <- reactive({
    tree <- rpart(cv_disease ~ trestbps + age + chol + fbs + restecg + cp, 
      data = filtered(), control = rpart.control(cp = 0.02), method = "class", x= TRUE)
    #tree$frame$yval <- as.numeric(rownames(tree$frame))
    return(tree)
  }) 
  
# reactive expressions to complete cross validation on the data "filtered()" for internal evaluation, also using rpart. 
  kfolds <- reactive ({ 

    # declare empty vectors for preformance stats and folds for k-fold corss validation
    F1.st <- c()
    precision.st <- c()
    recall.st <- c()
    k<-10
    folds <- createFolds(filtered()$cv_disease, k=k, list=TRUE, returnTrain = TRUE)#folds by caret package 
    
    #iterate rpart trees for folds
    for (i in 1:k) {
      xtree <- rpart(
        cv_disease ~  age + restecg + trestbps + chol + fbs + cp, 
        data = filtered()[folds[[i]],], method = "class")
      predictions <- predict(object = xtree, newdata = filtered()[-folds[[i]],], type = "class")
      F1.st<- c(F1.st,confusionMatrix(predictions, filtered()[-folds[[i]], ]$cv_disease,positive = ">50% Blockage")$byClass[c("F1")])
      precision.st <- c(precision.st,confusionMatrix(predictions, filtered()[-folds[[i]], ]$cv_disease,positive = ">50% Blockage")$byClass[c("Precision")])
      recall.st <- c(recall.st,confusionMatrix(predictions, filtered()[-folds[[i]], ]$cv_disease,positive = ">50% Blockage")$byClass[c("Recall")])
    }  
    # compose object for return containing performance stats. 
    v <- data.frame(F1.st,precision.st,recall.st)
    return(v) 
    
})

#Expression for creating a data frame that contains the results of k-fold cross validation.
  kstats <- reactive ({
    x <- data.frame(F1=double(),Precision = double,Recall = double())
    y <- data.frame(F1=double(),Precision = double,Recall = double())
    F1.st <- c()
    precision.st <- c()
    recall.st <- c()
    
    #replicate kfolds() 500 random times. 
    x<-replicate(500,kfolds())
    
    #combine results into a single object containing all kfold cv results
    for (j in 1 : 500) { 
      y <- rbind(y,x[,j])
    }
    
    #compose data frame into a single means and sd for each stat
    data.frame(F1 = mean(y$F1.st,na.rm = TRUE),
               Precision = mean(y$precision.st,na.rm = TRUE),
               Recall = mean(y$recall.st,na.rm = TRUE),
               SDF1 = sd(y$F1.st,na.rm = TRUE),
               SDP = sd(y$precision.st,na.rm = TRUE),
               SDR = sd(y$recall.st,na.rm = TRUE),  
               stringsAsFactors=FALSE) 
  })
  
# Output the expressions for the above cross- validation statistics
    output$validation <- renderText({
      temp <- kstats()
      #HTML for UI
      paste("<h3>Internal Model Validity  based on Source Selections:</h3><br>The <span style=\"font-weight:bold\">F1 score</span>, which is a weighted indicator of overall accuracy, is<span style=\"font-weight:bold\">",format(round(temp["F1"],2), nsmall = 2),"±",format(round((1.96*temp["SDF1"]),2), nsmall = 2),"</span><br>",
      "The <span style=\"font-weight:bold\">Recall </span>(Sens.), the proportion of true blockages correctly predicted, is<span style=\"font-weight:bold\">",format(round(temp["Recall"],2), nsmall = 2),"±",format(round((1.96*temp["SDR"]),2), nsmall = 2),"</span><br>",
      "The <span style=\"font-weight:bold\">Precision </span>(PPV), the proportion a blockage predictions that are true, is<span style=\"font-weight:bold\">",format(round(temp["Precision"],2), nsmall = 2),"±",format(round((1.96*temp["SDP"]),2), nsmall = 2),"</span><br>",
      "<p>*95% Confidence Intervals from repeated k-fold cross validations where n=500 and k=10<p>")
    })

#output the expression for any cross validation warning
  output$valid <- renderText({
    
    #declare default text "text1"
     text1 <- "<span style=\"font-size:150%;font-weight:bold\">Model shows an internal predictive validity.</span>"
    #declare warning text "text2" 
     text2 <- ""
     
    # call desired data for conditional 
     temp <- kstats()
     temprecall <- temp["Recall"]
     tempsd <- temp["SDR"]
     
     # conditional to determine the displayed text 
    if (temprecall > 0.94) {
      text1 <- ""
      text2 <- paste(text2,"<span style=\"color:red;font-size:150%;font-weight:bold\">🚑 Warning: High Recall.</span><br> Model created from very high prevalence source data and inadequate for predicting healthy patients.<br>")
    } 
     if (temprecall < 0.04) {
       text1 <- ""
       text2 <- paste(text2,"<span style=\"color:red;font-size:150%;font-weight:bold\">🚑 Warning: No Recall.</span><br> Model created from very low prevalence source data and is inadequate for predictions of diseased patients.<br>")
     } 
     if(temprecall < 0.55) {
      text1 <- ""
      text2 <- paste(text2,"<span style=\"color:red;font-size:150%;font-weight:bold\">📊 Warning: Low Recall.</span><br> In cross-validation, the model preforms at around random chance or worse. This may be caused by a large imbalance of disease and no disease, and low numbers in the source selection, which may create imbalanced cross validation folds, that models the complete source poorly.<br>")
    } 
     if(tempsd > 0.20) {
      text1 <- ""
      text2 <- paste(text2,"<span style=\"color:red;font-size:150%;font-weight:bold\">📊 Warning: High Variation in Cross-Validation.</span><br> Use with caution. May not be enough similar individuals to preform cross validation without imbalanced folds.<br>")
     }
     return(paste(text1,text2))
     
  })
  

  
# Output for the primary model function: use rtree object from rpart and output the answer.
  output$prediction <- renderText({
   
    if (test()$cp == "" || test()$fbs == "" ||test()$restecg == "" ){return(paste("<div style=\"color:red;font-size:115%\">Enter all 6 pretest attributes for the candidate patient to make a prediction.</div>"))}
   
    pred <- data.frame(predict(rtree(), test(), type = "prob"))
    #print(pred)
    pred <- format(round(pred[1,1], 2), nsmall = 2)
    if (pred <= 0.2) {
    pred <- toString(pred)
    pred <- paste("<span style=\"color:green;font-size:150%;font-weight: bold;\">💚 Very low likelihood of blockage.</span><br>The recursive partition model predicts only",pred,"probability of a coronary artery blockage of greater than 50%.", sep = " ")
    return(pred)
    }else if ((pred < 0.5) && (pred > 0.2)) {
      pred <- toString(pred)
      pred <- paste("<span style=\"color:orange;font-size:150%;font-weight: bold;\">💛 Low likelihood of a blockage.</span><br>The recursive partition model predicts only",pred,"probability of a coronary artery blockage of greater than 50%.", sep = " ")
      return(pred)
    }else if ((pred >= 0.5) && (pred < 0.8)) {
      pred <- toString(pred)
      pred <- paste("<span style=\"color:red;font-size:150%;font-weight: bold;\">❤️ Elevated probability of a blockage.</span><br>The recursive partition model predicts ",pred,"probability of a coronary artery blockage of greater than 50%.", sep = " ")
      return(pred)
    }else if (pred >= 0.8) {
      pred <- toString(pred)
      pred <- paste("<span style=\"color:darkred;font-size:150%;font-weight: bold;\">🖤 High probability of a blockage.</span><br>The recursive partition model predicts ",pred,"probability of a coronary artery blockage of greater than 50%.", sep = " ")
      return(pred)
    }else {return("Additional patient information needed for prediction")}
    })
  
# Show the ui provided test patient values using an HTML table
 output$values <- renderTable({
    #remove an extra column
    drops<-names(test()) %in% c("mean.of.chol.bps")
    test()[!drops]
   })
  
# Define the subset of data used for the x,y dot plot
  selectedData <- reactive({
    filtered()[c("age","mean.of.chol.bps")]
  })

# define the subset of data used for the dot color
  color <- reactive({
    filtered()[c("color_logic","line_logic")]
  })
  
# Create the x,y dot plot of the filtered() training data
  output$plot1 <- renderPlot({
    #colors
    palette(c("#e1e1e1","#DE7480","#555555", "#888888"))
    #margins
    par(mar = c(5.1, 4.1, 1, 1))
    #training points
    plot(selectedData(),
         lwd = 0.5,
         bg = color()$color_logic,
         col = "#222222",
         pch = 21,
         cex = 1.9,
         cex.axis = 1.2,
         cex.lab = 1.4,
         xlab = "Age",
         ylab = "Average of BP and Cholesterol")
    #test points
    points(test()[c("age","mean.of.chol.bps")], col = "#428BCA", bg = "white", pch = 21, cex = 2.2, lwd = 9)
    legend("topright", legend = c("(-)cath","(+)cath","candidate"), pch = c(21,21,21), col = c("black","black", "#428BCA"),
           cex = 1, pt.cex = 2 , pt.bg=c("#e1e1e1","#DE7480","white"), pt.lwd= c(0.5,0.5,8) ,ncol=2)
  })
  
# Create the rpart.plot for the decision tree model
  output$plot2 <- renderPlot({
    rpart.plot(rtree(), type = 4, extra = 108, xflip = TRUE, compress=TRUE, split.col="black", split.cex = 1.2, branch.col = "darkgray", under.col= "darkgray", leaf.round=0, round=0, box.palette = "RdYlGn")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

