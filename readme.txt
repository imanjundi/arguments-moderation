## Annotated sample of data from RegulationRoom for the paper 'Predicting Moderation of Deliberative Arguments:Is Argument Quality the Key?' (Neele Falk, Iman Jundi, Eva Maria Vecchi, Gabriella Lapesa)


We annotated a sample of 112 comments from RegulationRoom. 56 comments were moderated and the moderator intervention was annotated with 'improve comment quality' by Park et al (2012). Half of the comments were sampled from the lower quartile of the 'overall quality' score automatically annotated with a classifier by Lauscher et al. (2020). The automatic score range from 1 to 5 (1 meaning low quality). The data was annotators by 4 annotators with one aggregated score for argument quality following the guidelines. Their score range from 0(low) to 5 (high quality).

The sample contains the following columns:
UNIQUEID: a unique ID for the comment, a combination of the original RULE and an integer ID
MODERATED: True if the comment triggered a moderator intervention, False otherwise
OVERALL_QUAL: the automataically annotated score, overall quality
EFFECTIVE_QUAL: automatically annotated score for the effectiveness dimension
REASON_QUAL: automatically annotated score the reasonableness dimension
COGENCY_QUAL: automatically annotated score for the cogency dimension
CLEANEDCOMMENT: the content of the comment, URL and time stamps removed
annotator1: manually annotated score by annotator 1
annotator2: manually annotated score by annotator 2
annotator3:	manually annotated score by annotator 3
annotator4:	manually annotated score by annotator 4
averageannot: the average score of all annotators