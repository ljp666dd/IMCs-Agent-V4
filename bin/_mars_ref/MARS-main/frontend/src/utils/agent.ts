import { reactive } from "vue";
interface ViolationBOType {
    [key: string]: string;
}
const Agent: ViolationBOType = reactive({
    // 1
    Converter_Group_Admin: "Converter: Converter_Group_Admin",
    scheme_converter: "Converter: scheme_converter",
    converter_critic: "Converter: converter_critic",
    mergrid_ploter: "Converter: mergrid_ploter",
    scheme_code_writer: "Converter: scheme_code_writer",
    scheme_code_critic: "Converter: scheme_code_critic",
    // 2

    expriment_code_writer: "Executor: expriment_code_writer",
    data_collector: "Executor: data_collector",
    collector_code_writer: "Executor: collector_code_writer",
    Inner_Executor_Admin: "Executor: Inner_Executor_Admin",
    // Inner_Executor_Admin: "Executor: Executor_Group_Admin",
    // 3
    Generate_Group_Admin: "Generate: Generate_Group_Admin",
    structure_scientist: "Generate: structure_scientist",
    property_scientist: "Generate: property_scientist",
    application_scientist: "Generate: application_scientist",
    synthesis_scientist: "Generate: synthesis_scientist",
    scheme_critic: "Generate: scheme_critic",

    // 4
    analysis_executor: "Optimize: analysis_executor",
    analysis_pl_uv: "Optimize: analysis_pl_uv",
    analysis_picturer: "Optimize: analysis_picturer",
    Experiment_Optimizer: "Optimize: Experiment_Optimizer",
    optimizer_critic: "Optimize: optimizer_critic",
    Analysis_Group_Admin: "Optimize: Analysis_Group_Admin",

    // 
    // Outer_Retrieval_Admin: "Retrieval: Outer_Retrieval_Admin",
    // Outer_Converter_Admin: "Converter: Outer_Converter_Admin",
    // Outer_Executor_Admin: "Executor: Outer_Executor_Admin",
    experiment_executor: "Executor: experiment_executor",
    // Outer_Generate_Admin: "Generate: Outer_Generate_Admin",
    // Outer_Analysis_Admin: "Optimize: Outer_Analysis_Admin",
    // vector_code_executor: "Retrieval: vector_code_executor",
    // graphrag_code_executor: "Retrieval: graphrag_code_executor",
    // web_code_executor: "Retrieval: web_code_executor",
    // web_summary: "Retrieval: web_summary",

    // 5
    Inner_Retrieval_Admin: "Retrieval: Inner_Retrieval_Admin",
    // Inner_Retrieval_Admin: "Retrieval: Retrieval_Group_Admin",
    vector_searcher: "Retrieval: vector_searcher",
    graphrag_searcher: "Retrieval: graphrag_searcher",
    web_searcher: "Retrieval: web_searcher",
    Planer: "Planner",
    Planner: "Planner",
})
export function getAgent(data: any) {
    let rData = Object.keys(Agent).filter(k => k.toLowerCase() === data.toLowerCase())[0]
    return Agent[rData] || ''
}