using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xamasoft.JsonClassGenerator.Utils;

namespace Xamasoft.JsonClassGenerator.CodeWriters
{
    public class JavaCodeWriter : ICodeWriter
    {
        public string FileExtension
        {
            get { return ".java"; }
        }

        public string DisplayName
        {
            get { return "Java"; }
        }

        public string GetTypeName(JsonType type, IJsonClassGeneratorConfig config)
        {
            var arraysAsLists = !config.ExplicitDeserialization;

            switch (type.Type)
            {
                case JsonTypeEnum.Anything: return "Object";
                case JsonTypeEnum.Array: return arraysAsLists ? "IList<" + GetTypeName(type.InternalType, config) + ">" : GetTypeName(type.InternalType, config) + "[]";
                case JsonTypeEnum.Dictionary: return "Map<string, " + GetTypeName(type.InternalType, config) + ">";
                case JsonTypeEnum.Boolean: return "boolean";
                case JsonTypeEnum.Float: return "float";
                case JsonTypeEnum.Double: return "double";
                case JsonTypeEnum.Integer: return "int";
                case JsonTypeEnum.Long: return "long";
                case JsonTypeEnum.Date: return "Date";
                case JsonTypeEnum.NonConstrained: return "Object";
                case JsonTypeEnum.NullableBoolean: return "Boolean";
                case JsonTypeEnum.NullableFloat: return "Double";
                case JsonTypeEnum.NullableInteger: return "Integer";
                case JsonTypeEnum.NullableLong: return "Long";
                case JsonTypeEnum.NullableDate: return "Date";
                case JsonTypeEnum.NullableSomething: return "Object";
                case JsonTypeEnum.Object: return type.AssignedName;
                case JsonTypeEnum.String: return "String";
                default: throw new System.NotSupportedException("Unsupported json type");
            }
        }

        public void WriteClass(IJsonClassGeneratorConfig config, TextWriter sw, JsonType type)
        {
            sw.WriteLine("/**");
            sw.WriteLine(" * Description");
            sw.WriteLine(" * ");
            sw.WriteLine(" * @author Dincat@WorrilessGo");
            sw.WriteLine(" */");

            var visibility = config.InternalVisibility ? "internal" : "public";


            if (config.UseNestedClasses)
            {
                if (!type.IsRoot)
                {
                    sw.WriteLine("{0} class {1}", visibility, type.AssignedName);
                    sw.WriteLine("{");
                }
            }
            else
            {
                sw.WriteLine("{0} class {1}", visibility, type.AssignedName);
                sw.WriteLine("{");
            }

            var prefix ="    ";

            WriteClassMembers(config, sw, type, prefix);


            if (config.UseNestedClasses && !type.IsRoot)
                sw.WriteLine("}");

            if (!config.UseNestedClasses)
                sw.WriteLine("}");

            //sw.WriteLine();
        }

        public void WriteFileStart(IJsonClassGeneratorConfig config, TextWriter sw)
        {
           
        }

        public void WriteFileEnd(IJsonClassGeneratorConfig config, TextWriter sw)
        {
            if (config.UseNestedClasses)
            {
                sw.WriteLine("}");
            }
        }

        public void WriteNamespaceStart(IJsonClassGeneratorConfig config, TextWriter sw, bool root)
        {
            if (string.IsNullOrEmpty(config.Namespace)) 
            {
                return;
            }

            sw.WriteLine();
            sw.WriteLine("package ",  config.Namespace);
            sw.WriteLine();
        }

        public void WriteNamespaceEnd(IJsonClassGeneratorConfig config, TextWriter sw, bool root)
        {
           
        }


        private void WriteClassMembers(IJsonClassGeneratorConfig config, TextWriter sw, JsonType type, string prefix)
        {
            IList<FieldInfo> theFields = type.Fields;
            if (config.SortMemberFields) theFields = theFields.OrderBy(f => f.JsonMemberName).ToList();
            foreach (var field in theFields)
            {
                if (config.UsePascalCase || config.ExamplesInDocumentation) sw.WriteLine();

                if (config.ExamplesInDocumentation)
                {
                    sw.WriteLine(prefix + "/**");
                    sw.WriteLine(prefix + " * Examples: " + field.GetExamplesText());
                    sw.WriteLine(prefix + "*/");
                }

                if (config.UsePascalCase)
                {
                    sw.WriteLine(prefix + "@JsonProperty(value=\"{0}\")", field.JsonMemberName);
                }

                var lowCaseName = StringUtils.LowerCaseFirstLetter(field.MemberName);
                sw.WriteLine(prefix + "private {0} {1};", field.Type.GetTypeName(), lowCaseName);
                sw.WriteLine();
                sw.WriteLine(prefix + "public {0} get{1}(){{ return {2}; }}", field.Type.GetTypeName(), field.MemberName, lowCaseName);
                sw.WriteLine();
                sw.WriteLine(prefix + "private {0} set{1}({2}){{ this.{3}={4} }}", field.Type.GetTypeName(), field.MemberName, lowCaseName, lowCaseName, lowCaseName);

                
            }

        }

    }
}
