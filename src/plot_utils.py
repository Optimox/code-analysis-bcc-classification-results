from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import plotly.express as px
from CI_utils import sensitivity_and_specificity_with_confidence_intervals
import pandas as pd
import plotly.graph_objects as go

def generate_auc_fig(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
    fig_auc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc_score:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_auc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_auc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc.update_xaxes(constrain='domain')
    # fig_auc.show()

    # disabled color area
    fig_auc.data[0].stackgroup =""
    # switch color to green
    fig_auc.data[0].line = {'color': '#248a19'}
    return fig_auc, auc_score

def add_inset(fig, inset_fig):
    # Define the domain for the inset plot
    inset_x_domain = [0.35, 0.98]  # for x-axis on the bottom right
    inset_y_domain = [0.05, 0.65]  # for y-axis on the bottom right
    # Add new axes to the main figure for the inset plot
    fig.update_layout(
        xaxis2=dict(
            domain=inset_x_domain,
            anchor='y2',
        ),
        yaxis2=dict(
            domain=inset_y_domain,
            anchor='x2',
        )
    )
    
        
    # Adding each trace from the inset to the main figure, specifying the new axes
    for trace in inset_fig.data:
        new_trace = trace.update(xaxis='x2', yaxis='y2')  # Assign the trace to the inset axes
        new_trace.showlegend = False  # Hide the trace from the legend
        fig.add_trace(new_trace)
    
    fig.add_annotation(
        x=-0.01,  # Starting from the left; adjust as necessary for exact positioning
        y=1.02,  # Starting from the top; adjust as necessary
        xref="x2",  # Reference to the inset's x-axis
        yref="y2",  # Reference to the inset's y-axis
        text="Zoom in",  # The text to display
        showarrow=False,  # Don't show an arrow pointing to the text
        font=dict(
            size=12,  # Adjust the size of the text as necessary
            color="black",  # Set the color of the text
        ),
        align="left",  # Align the text to the left
        yanchor="top",  # Anchor the text to the top for y positioning
        xanchor="left",  # Anchor the text to the left for x positioning
        bordercolor="black",  # Optional: add a border color
        borderwidth=1,  # Optional: set border width
        bgcolor="white",  # Optional: set a background color to make the text stand out
        opacity=0.7  # Optional: set opacity for the background color
    )

    
    # Now add a rectangle around the inset plot
    fig.add_shape(
        # Rectangle reference to the axes
        type="rect",
        xref="paper", yref="paper",
        x0=inset_x_domain[0], y0=inset_y_domain[0],  # Lower left corner
        x1=inset_x_domain[1], y1=inset_y_domain[1],  # Upper right corner
        line=dict(
            color="Black",
            width=2,
            dash="dashdot",
        ),
        layer="above",
    )
#     fig.update_layout(showlegend=False)
    return fig

def generate_confidence_plot(df_results, df_ai_scores, df_answers, sens_spec_df_ai_2, sens_spec_df_no_ai_2, alpha, size=None, showlegend=False, zoom_range=[-0.02, 1.02]):
    fig_auc, auc_score = generate_auc_fig(y_true=df_ai_scores.is_diagnostic_bcc, y_score=df_ai_scores.max_moving_avg_24)
   
    color_pairs = [
        ('#1F77B4', '#D62728'),
        ('#17BECF', '#E377C2'),
        ('#306BAC', '#FC4E2A')
    ]

    dict_markers = {"experts": "star",
                    "intermediates": "cross",
                    "novices": 'triangle-up'
                   }
    
    df_errors_groups = []
    for (user_type, type_), df_user_type in df_answers.groupby(["user_type", "ai_assistance_present"]):
        tn, fp, fn, tp = confusion_matrix(y_true=df_user_type.is_diagnostic_bcc.values.astype(float),
                         y_pred=df_user_type.lcoct_phase_answer.values.astype(float)).ravel()

        sensitivity_point_estimate, specificity_point_estimate, \
            sensitivity_confidence_interval, specificity_confidence_interval \
            = sensitivity_and_specificity_with_confidence_intervals(TP=tp, FP=fp, FN=fn, TN=tn, alpha=alpha)
        df_errors_groups.append({"user_type":user_type,
                                  "ai_assistance_present":type_,
                                  # spec needs to be inversed because we are looking at 1 - spec
                                  "spec_error_plus" : specificity_point_estimate - specificity_confidence_interval[0],
                                  "spec_error_minus" : specificity_confidence_interval[1] - specificity_point_estimate,
                                  "sens_error_plus" : sensitivity_confidence_interval[1] - sensitivity_point_estimate,
                                  "sens_error_minus" : sensitivity_point_estimate - sensitivity_confidence_interval[0],
                                 })
    df_errors_groups = []
    for (user_type, type_), df_user_type in df_answers.groupby(["user_type", "ai_assistance_present"]):
        tn, fp, fn, tp = confusion_matrix(y_true=df_user_type.is_diagnostic_bcc.values.astype(float),
                         y_pred=df_user_type.lcoct_phase_answer.values.astype(float)).ravel()

        sensitivity_point_estimate, specificity_point_estimate, \
            sensitivity_confidence_interval, specificity_confidence_interval \
            = sensitivity_and_specificity_with_confidence_intervals(TP=tp, FP=fp, FN=fn, TN=tn, alpha=alpha)

        df_errors_groups.append({"user_type":user_type,
                          "ai_assistance_present":type_,
                          # spec needs to be inversed because we are looking at 1 - spec
                          "spec_error_plus" : specificity_point_estimate - specificity_confidence_interval[0],
                          "spec_error_minus" : specificity_confidence_interval[1] - specificity_point_estimate,
                          "sens_error_plus" : sensitivity_confidence_interval[1] - sensitivity_point_estimate,
                          "sens_error_minus" : sensitivity_point_estimate - sensitivity_confidence_interval[0],
                         })
    df_errors_groups = pd.DataFrame(df_errors_groups)


    sens_spec_2 = pd.concat([sens_spec_df_ai_2, sens_spec_df_no_ai_2]).reset_index()
    sens_spec_2["1-specificity"] = 1-sens_spec_2["specificity"]
    nb_bcc_groups = df_answers.groupby(["user_type", "ai_assistance_present"]).is_diagnostic_bcc.sum().reset_index(drop=False)
    sens_spec_2 = pd.merge(sens_spec_2, nb_bcc_groups, on=["user_type", "ai_assistance_present"], how="inner").drop("index", axis=1)
    sens_spec_2 = pd.merge(sens_spec_2, df_errors_groups, on=["user_type", "ai_assistance_present"], how="inner")
    sens_spec_2[" "] = sens_spec_2["ai_assistance_present"]
    print(sens_spec_2.shape, sens_spec_2.columns)
    fig = px.scatter(sens_spec_2, x="1-specificity", y="sensitivity", color=" ", 
                       text="user_type", # disable to mask names
                     error_x="spec_error_plus", error_x_minus="spec_error_minus",
                     error_y="sens_error_plus", error_y_minus="sens_error_minus", opacity=0.5
#                 title="Comparison of different expertise groups with (blue) and without AI help (red). \n In green, the AI ROC curve.",

                    )
    fig.update_traces(textposition="top right", showlegend=False, textfont=dict(size=15))
#     for trace in fig.data:
#         trace.showlegend = False
#         trace.name = ''
#     for trace in fig.data:
    
    for user, user_df in sens_spec_2.groupby("user_type"):
        user_df = user_df.reset_index(drop=True)
        fig.add_shape(
            type='line', line=dict(dash='dash', color="black"),
            name=user+str(showlegend),
            x0=user_df["1-specificity"].values[0], x1=user_df["1-specificity"].values[1],
            y0=user_df["sensitivity"].values[0], y1=user_df["sensitivity"].values[1],
            xref='x2', yref='y2',  # Draw this line on the inset plot
        )
        
    # Define the name for the AUC trace
    auc_trace_name = f'AI ROC Curve (AUC={auc_score:.4f})'
    # Update the AUC trace to have a legend entry and the desired appearance
    fig_auc.data[0].update(
        line=dict(color='#248a19',# dash='dash'
                 ),
        name=auc_trace_name,
        showlegend=True
    )
    # Add the updated AUC trace to your main figure
    fig.add_trace(fig_auc.data[0])
    
    # Optionally, customize the legend's appearance and position
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.90
    ))
    
    
    # fig.add_trace(fig_auc.data[0])

    for user_type_idx, user_type in enumerate(df_results.user_type.unique()):
        df_res_ai = df_results[(df_results["clinical_phase"]==False) & (df_results["ai_help"]==True) & (df_results["user_type"]==user_type)].reset_index(drop=True)
        fig.add_trace(
            go.Scatter(
                x=df_res_ai["1-specificity"].values,
                y=df_res_ai["sensitivity"].values,
                hovertext=df_res_ai["user"].values,
                mode='markers',
                name=f'{user_type} with ai ',
                marker=dict(
                    symbol=dict_markers[user_type],  # specify the symbol you want for the markers
    #                 size=8,
                    color="blue", #color_pairs[user_type_idx][0]  # specify the color you want for the markers
                ),
            )
        )

        df_res_no_ai = df_results[(df_results["clinical_phase"]==False)& (df_results["ai_help"]==False)  & (df_results["user_type"]==user_type)].reset_index(drop=True)
        fig.add_trace(
            go.Scatter(
                x=df_res_no_ai["1-specificity"].values,
                y=df_res_no_ai["sensitivity"].values,
                hovertext=df_res_no_ai["user"].values,
                mode='markers',
                name=f'{user_type} without ai',
                marker=dict(
                    symbol=dict_markers[user_type],
                    color="red", #color_pairs[user_type_idx][1]  # specify the color you want for the markers
                ),
            )
        )

    fig.update_xaxes(range=zoom_range)  # Limit x-axis from 0 to 1
    fig.update_yaxes(range=zoom_range)  # Limit y-axis from 0 to 1

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_xaxes(
        constrain='domain'  # Ensures that the scale of the x-axis is constrained to the 'domain' which helps in maintaining aspect ratio
    )
    fig.update_layout(
        legend=dict(
            bgcolor='rgba(255,255,255,0)',  # Transparent background
    #         bordercolor='black',             # Black border
    #         borderwidth= 2                   # Border width in pixels
        )
    )
    
    if size is not None:
        # Set the size of the figure
        fig.update_layout(
            width=size,  # Width in pixels
            height=size,  # Height in pixels
        #     autosize=False  # This disables autosizing to use the specified width and height
        )

    # uncomment to disable legend
    fig.update_layout(showlegend=False)

    
    
    # add random baseline
    fig.add_shape(
        type='line', line=dict(dash='dash', color="blue"),
        x0=0, x1=0.34, y0=0, y1=0.34,
    )

    fig.add_shape(
        type='line', line=dict(dash='dash', color="blue",),
        x0=0.66, x1=0.75, y0=0.66, y1=0.75,
    )

    fig.add_shape(
        type='line', line=dict(dash='dash', color="blue",), 
        x0=0.91, x1=1, y0=0.91, y1=1,
    )
    
    fig.update_layout(showlegend=showlegend)
    
    
    # Now add a rectangle around the inset plot
    # rectangle
    fig.add_shape(
        # Rectangle reference to the axes
        type="rect",
        xref="paper", yref="paper",
        x0=0.0, y0=0.64,  # Lower left corner
        x1=0.32, y1=1.,  # SPECIFICTY x
        line=dict(
            color="Black",
            width=2,
            dash="dashdot",
        ),
        layer="above",
    )
    
    fig.add_trace(
    go.Scatter(
        x=[0],  # X coordinates for the stars
        y=[1],  # Y coordinates for the stars
        name="Perfect score",
        mode='markers',  # Combine markers and text
        marker=dict(
            symbol='star',  # Use a star symbol
            size=12,  # Adjust size as needed
            color='orange',  # Star color
        ),
        text=['Perfect score'],  # Text labels
        textposition='top right',  # Position text above the markers
        showlegend=True  # Ensure these markers don't appear in the legend
    )
)
    
    
    return fig



