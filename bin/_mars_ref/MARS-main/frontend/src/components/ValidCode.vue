<template>
    <div class="CharacterVerification" :style="{'width' : width + 'px','height' : height + 'px'}" ref="CharacterVerification">
      <canvas
        @click="refresh"
        :width="width"
        :height="height"
        ref="verifyCanvas"
        :style="{ cursor: 'pointer' }"
      ></canvas>
    </div>
  </template>
   
  <script lang="ts">
  import { defineComponent, ref, onMounted } from "vue";
  export default defineComponent({
    name: "CharacterVerification",
    props: {
      width: {
        type: Number,
        default: 150,
      },
      height: {
        type: Number,
        default: 40,
      },
      type: {
        type: String,
        default: "number", //图形验证码默认类型blend:数字字母混合类型、number:纯数字、letter:纯字母
      },
    },
    setup(props, {expose}) {
      const numArr = "0,1,2,3,4,5,6,7,8,9".split(",");
      const letterArr =
        "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(
          ","
        );
      const code = ref("");
      const CharacterVerification = ref<HTMLDivElement | null>(null);
      const verifyCanvas = ref<HTMLCanvasElement | null>(null);
      let ctx;
      onMounted(() => {
        refresh();
      })
      const refresh = () => {
        code.value = "";
        if (verifyCanvas.value?.getContext) {
          ctx = verifyCanvas.value.getContext("2d") as CanvasRenderingContext2D;
        } else {
          return;
        }
   
        ctx.textBaseline = "middle";
   
        ctx.fillStyle = randomColor(180, 240);
        ctx.fillRect(0, 0, props.width, props.height);
   
        if (props.type == "blend") {
          var txtArr = numArr.concat(letterArr);
        } else if (props.type == "number") {
          var txtArr = numArr;
        } else {
          var txtArr = letterArr;
        }
   
        for (var i = 1; i <= 4; i++) {
          var txt = txtArr[randomNum(0, txtArr.length)];
          code.value += txt;
          ctx.font = randomNum(props.height / 2, props.height) + "px SimHei"; //随机生成字体大小
          ctx.fillStyle = randomColor(50, 160); 
          ctx.shadowOffsetX = randomNum(-3, 3);
          ctx.shadowOffsetY = randomNum(-3, 3);
          ctx.shadowBlur = randomNum(-3, 3);
          ctx.shadowColor = "rgba(0, 0, 0, 0.3)";
          var x = (props.width / 5) * i;
          var y = props.height / 2;
          var deg = randomNum(-30, 30);
          ctx.translate(x, y);
          ctx.rotate((deg * Math.PI) / 180);
          ctx.fillText(txt, 0, 0);
          ctx.rotate((-deg * Math.PI) / 180);
          ctx.translate(-x, -y);
        }
        for (var i = 0; i < 4; i++) {
          ctx.strokeStyle = randomColor(40, 180);
          ctx.beginPath();
          ctx.moveTo(randomNum(0, props.width), randomNum(0, props.height));
          ctx.lineTo(randomNum(0, props.width), randomNum(0, props.height));
          ctx.stroke();
        }
        for (var i = 0; i < props.width / 4; i++) {
          ctx.fillStyle = randomColor(0, 255);
          ctx.beginPath();
          ctx.arc(
            randomNum(0, props.width),
            randomNum(0, props.height),
            1,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
      };
   
      const randomColor = (min: number, max: number) => {
        var r = randomNum(min, max);
        var g = randomNum(min, max);
        var b = randomNum(min, max);
        return "rgb(" + r + "," + g + "," + b + ")";
      };
      const randomNum = (min: number, max: number) => {
        return Math.floor(Math.random() * (max - min) + min);
      };
      const validate = (iptCode: string) => {
        var newIptCode = iptCode.toLowerCase();
        var v_code = code.value.toLowerCase();
        if (newIptCode == v_code) {
          return true;
        } else {
          refresh();
          return false;
        }
      };
   
      expose({ validate });
   
      return {
        CharacterVerification,
        verifyCanvas,
        refresh,
      };
    },
  });
  </script>
   
  <style lang="scss" scoped>
  </style>