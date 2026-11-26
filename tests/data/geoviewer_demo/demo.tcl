######################################################################
#
# Macro Name    : dem/v4_demos/surpac_demo/demo.tcl
#
# Version       : Surpac Vision V4.1-G 
#
# Creation Date : Thu Nov 01 12:21:58 2001
#
# Description   : 
#
######################################################################

set status [ SclFunction "RECALL ANY FILE" {
  file="project.swa"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="surpac.ddb"
  mode="none"
}]

set status [ SclFunction "CHECK DATABASE CONNECTED" {} ]

set status [ SclFunction "DRILLHOLE ATTRIBUTES" {
  trace_colour="forest green"
  trace_thickness="1"
  legend_location="no legend"
  table="geology"
  field="rock"
  tick_mark_weight="1"
  rescale="true"
  add_constraint="true"
  frm00012={
    {
      cons={
        {
          field=""
          operator="="
          value=""
        }
      }
    }
  }
}]

set status [ SclFunction "DRILLHOLE COLLARS" {
  show_collars="true"
  collar_marker="O"
  collar_marker_size="0.5"
  position="along path"
  num_rows="0"
}]

set status [ SclFunction "DRILLHOLE GEOLOGY" {
  hide="false"
  num_rows="0"
}]

set status [ SclFunction "DRILLHOLE LABELS" {
  hide="true"
  tick_length="1"
  num_rows="0"
}]

set status [ SclFunction "DRILLHOLE GRAPHS" {
  hide="false"
  num_rows="0"
}]

set status [ SclFunction "DRILLHOLE DEPTH MARKERS" {
  position="left"
  offset="5"
  decimals="2"
  use_styles="false"
  colour=""
  font_style="roman"
  font_size="1"
  font_units="oru"
  alignment="right"
  major="10"
  minor="0"
  display="false"
  major_length="3"
  minor_length="1"
  hide="false"
}]

set status [ SclFunction "DRILLHOLE APPARENT DIP MARKERS" {
  hide="false"
  num_rows="0"
}]

set status [ SclFunction "SLICING PLANES" {
  frm00298={
    {
      width="100"
      method="Northing"
      plane_thickness="20"
      step_distance="40"
      value="7120,7560,40"
      sectionBlockModel="N"
      block_layer="block sections"
      tolerance="0"
      section_objects="Y"
      object_range=""
      object_layer="object sections"
      confirm_line_ends="N"
      interval_range="range"
      colour_fill_bm_sections="N"
      zoom_plane="Y"
      view_limits="Y"
    }
  }
}]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "VIEWING PLANES FORWARD" {} ]

set status [ SclFunction "CHECK SECTION FOR SAVING" {} ]

set status [ SclFunction "SAVE DH SECTIONS TO STRINGS" {
  DrillholeLocation="d"
  BlockmodelLocation=""
  ObjectLocation="o"
  CurrentGraphicsLocation=""
  SectionRange="7120,7560,40"
  RealWorldCoords="false"
  DownholeDepths="5"
}]

set status [ SclFunction "GEOLOGY DATABASE DISCONNECT" {} ]

set status [ SclFunction "EXIT GRAPHICS" {
  frm00108={
    {
      exit_mode="E"
    }
  }
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7120.str"
  mode="none"
}]

set status [ SclFunction "IDENTIFY POINT" {
  select_point={
    {
      winx="-0.056"
      winy="-0.151"
      objectx="1491.461"
      objecty="98.487"
      objectz="7119.489"
    }
  }
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7120.str"
  mode="none"
}]

set status [ SclFunction "IDENTIFY POINT" {
  select_point={
    {
      winx="-0.393"
      winy="-0.509"
      objectx="1472.160"
      objecty="-146.777"
      objectz="7200.000"
    }
  }
}]

set status [ SclFunction "SECTION" {} ]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7120.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7160.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7200.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7240.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7280.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7320.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7360.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7400.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7440.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7480.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7520.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="o7560.str"
  mode="none"
}]

set status [ SclFunction "SECTION" {} ]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7120.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7160.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7200.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7240.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7280.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7320.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7360.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7400.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7440.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7480.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7520.str"
  mode="none"
}]

set status [ SclFunction "RECALL ANY FILE" {
  file="d7560.str"
  mode="none"
}]

set status [ SclFunction "ZOOM ALL" {} ]

puts "Use the Viewing planes function (alias = VP) with a step size of 40 to view the individual sections"
puts "Move between sections using Viewing planes forward and Viwewing planes backward (alias VPF and VPB)"
puts "Use the Delete cutting plane function to restore the view"



