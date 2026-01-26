using System;
using System.Collections.Generic;

namespace zdhsys
{
    public class ExperimentData
    {
        public string TaskId { get; set; }
        public string ExperimentName { get; set; }
        public List<Material> Materials { get; set; }
        public List<Container> Containers { get; set; }
        public List<Equipment> Equipments { get; set; }
        public List<WorkflowStep> RobotWorkflow { get; set; }
    }

    public class Material
    {
        public string MaterialId { get; set; }
        public string Name { get; set; }
        public string Amount { get; set; }
        public string Unit { get; set; }
        public string Purity { get; set; }
        public string State { get; set; }
        public string Formula {get; set;}
    }

    public class Container
    {
        public string ContainerId { get; set; }
        public string Name { get; set; }
        public string Capacity { get; set; }
        public string Unit { get; set; }
        public string MaterialOfConstruction { get; set; }
        public string Shape { get; set; }
        public string HeatResistant { get; set; }
        public string PressureRating { get; set; }
    }

    public class Equipment
    {
        public string EquipmentId { get; set; }
        public string Name { get; set; }
        public Dictionary<string, string> Parameters { get; set; }
    }

    public class WorkflowStep
    {
        public string StepId { get; set; }
        public string Description { get; set; }
        public List<Action> Actions { get; set; }
        public List<string> Dependencies { get; set; }
        public StepOutput StepOutput { get; set; }
    }

    public class Action
    {
        public string ActionType { get; set; }
        public string ContainerId { get; set; }
        public string MaterialId { get; set; }
        public string EquipmentId { get; set; }
    }

    public class StepOutput
    {
        public string ContainerId { get; set; }
        public List<MaterialContent> Contents { get; set; }
    }

    public class MaterialContent
    {
        public string MaterialId { get; set; }
        public string Amount { get; set; }
        public string Unit { get; set; }
    }
}