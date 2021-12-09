# Gráfico paper AGRANDA 2021

library(readr)
library(dplyr)
library(reshape2)
library(ggplot2)
require(forcats)
library(gridExtra)
library(grid)




resultados_agranda_2021 <- read_csv("resultados-agranda-2021.csv")
resultados_agranda_2021 <- read_csv("f1_example.csv")
resultados_agranda_2021 <- read_csv("f1_micro.csv")
resultados_agranda_2021 <- read_csv("f1_macro.csv")

data = melt(resultados_agranda_2021, id.vars = c("stream"))
names(data) = c("Stream", "Algoritmo", "F1")
data$tipo_var = ifelse(data$Algoritmo %in% c("Dominio","N","A","L","LC","LD"),"Metrica","Algoritmo")


data.alg = data[data$tipo_var=="Algoritmo",]

data.alg$F1 = as.numeric(data.alg$F1)
data.alg$Algoritmo = factor(data.alg$Algoritmo)
data.alg$Propuesta = ifelse(data.alg$Algoritmo %in% c("EFMP","EFMP2"),"Nuestra","Otras")


# data.alg$Algoritmo = factor(data.alg$Algoritmo,levels = c("Fri", "Sat", "Sun", "Thur"))
head(data.alg)
str(data.alg)

unique(data$Algoritmo)

# https://stackoverflow.com/questions/34001024/ggplot-order-bars-in-faceted-bar-chart-per-facet
names <- unique(data.alg$Stream)

plist <- list()
plist[]

for (i in 1:length(names)) {
  d <- subset(data.alg, Stream == names[i])
  d$Algoritmo <- factor(d$Algoritmo, levels=d[order(d$F1),]$Algoritmo)
  
  dd = data[data$Stream=="20ng" & data$tipo_var=="Metrica",c(2,3)]
  names(dd) = c("Medida", "Valor")
  # dd
  
  metricasDS = paste(
    paste("N:", data[data$Stream == names[i] & data$Algoritmo=="N",]$F1),
    paste("L:", data[data$Stream == names[i] & data$Algoritmo=="L",]$F1),
    paste("LC:", data[data$Stream == names[i] & data$Algoritmo=="LC",]$F1),
    paste("LD:", data[data$Stream == names[i] & data$Algoritmo=="LD",]$F1),
    sep="\n"
  )
  # print(metricasDS)
  # 
  p1 <- ggplot(d, aes(x = Algoritmo, y = F1, fill = Propuesta, width=0.75)) + 
    labs(y = "F1", x = NULL, fill = NULL) +
    geom_bar(stat = "identity",color="white", position=position_dodge()) +
    facet_wrap(~Stream) +
    scale_y_continuous(breaks=seq(0.0, 0.6, 0.1), limits = c(0,0.6)) +
    #coord_flip() +
    #guides(fill=FALSE) +
    theme_classic() + theme( strip.background  = element_blank(),
                        #panel.grid.major = element_line(colour = "grey80"),
                        panel.border = element_blank(),
                        axis.ticks = element_line(size = 0),
                        panel.grid.minor.y = element_blank(),
                        panel.grid.major.y = element_blank(), 
                        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
                        legend.position=c(.2,.6)
                        
                        ) +
    
    scale_fill_manual(values = c("Nuestra"="#3b3b3d","Otras" = "#ebccca")) +
    theme() +
    # annotation_custom(tableGrob(dd ,show.rownames=FALSE))
  
    #annotate(geom="text", x=1.5, y=0.5, hjust=0.5, label=metricasDS, color="black") 
    annotate(geom="text", x=1.5, xmin=1.5,xmax=2.5, y=0.6, hjust=0.06, label=paste("N:", data[data$Stream == names[i] & data$Algoritmo=="N",]$F1),
             color="black") +
    annotate(geom="text", x=1.5, xmin=1.5,xmax=2.5, y=0.55, hjust=0.06,label=paste("L:", data[data$Stream == names[i] & data$Algoritmo=="L",]$F1),
             color="black") +
    annotate(geom="text", x=1.5, xmin=1.5,xmax=2.5, y=0.5, hjust=0.1,label=paste("LC:", data[data$Stream == names[i] & data$Algoritmo=="LC",]$F1),
             color="black") +
    annotate(geom="text", x=1.5, xmin=1.5,xmax=2.5, y=0.45, hjust=0.1,label=paste("LC:", data[data$Stream == names[i] & data$Algoritmo=="LD",]$F1),
             color="black")

  plist[[names[i]]] = p1
}   

do.call("grid.arrange", c(plist, ncol=3))
        



c("BR" = "#ebccca",
  "CC" = "#ebccca",
  "MLHT" = "#ebccca",
  
  "DWM_BR" = "#ebccca",
  "DWM_CC" = "#ebccca",
  "ENS_BR" = "#ebccca",
  "ENS_CC" = "#ebccca",
  
  "EFMP" = "#3b3b3d",
  "EFMP2" = "#3b3b3d")

#install.packages("devtools")
#devtools::install_github("ricardo-bion/ggradar")
#library(ggradar)

#ggradar(df) 
