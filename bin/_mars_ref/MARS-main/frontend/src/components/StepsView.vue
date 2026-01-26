<template>
  <div class="list-box">
    <div
      class="item-box"
      v-for="(item, index) in props.reasoningList"
      :key="index"
    >
      <div class="title" @click="toTitle(index)">
        {{ item.title }}
      </div>
      <div class="ul">
        <div
          @click="toContent(index, num)"
          v-for="(key, num) in item.children"
          :key="num"
          style="width: 100%;"
        >
          <div v-if="key.source !== 'Planner'" class="li">
            <div
              class="p-box"
              :class="{
                active_item:
                  index === props.reasoningList.length - 1 &&
                  num === item.children.length - 1,
              }"
            >
              <img
                v-if="
                  index === props.reasoningList.length - 1 &&
                  num === item.children.length - 1
                "
                src="../assets/layout/vector.png"
                alt=""
              />
              <span>{{ `${key.source || ''}` }}</span>
              <!-- <span>{{ `${key.source?key.source+':':''} ${key.agent_name}` }}</span> -->
            </div>
            <img
              class="next"
              v-if="num + 1 != item.children.length"
              src="../assets/layout/next.png"
              alt=""
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { defineProps } from "vue";

const props: any = defineProps({
  reasoningList: {
    type: Array,
    default: [],
  },
});
const toTitle = (index: number) => {
  let id = `${index}title`;
  let view: any = document.getElementById(id);
  view.scrollIntoView({ behavior: "smooth" });
};
const toContent = (index: number, num: number) => {
  let id = `${index}${num}content`;
  let view: any = document.getElementById(id);
  view.scrollIntoView({ behavior: "smooth" });
};
</script>
<style scoped lang="less">
.list-box {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  row-gap: 18px;
  height: calc(100% - 70px);
  overflow: auto;
  &::-webkit-scrollbar {
    width: 0;
    height: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: #1a87ca;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #1a87ca;
  }
  .item-box {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-sizing: border-box;
    padding: 12px;
    position: relative;

    &::before {
      content: "";
      position: absolute;
      bottom: -12px;
      left: 50%;
      transform: translateX(-50%);
      width: 80%;
      height: 1px;
      background: linear-gradient(
        90deg,
        rgba(37, 237, 242, 0) 0%,
        rgba(98, 250, 248, 0.2) 32.5%,
        rgba(98, 250, 248, 0.2) 68%,
        rgba(37, 237, 242, 0) 100%
      );
    }
    &:last-child::before {
      height: 0;
    }
    .title {
      width: 100%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 15px;
      color: #fff;
      margin-bottom: 12px;
      cursor: pointer;
    }

    .ul {
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      row-gap: 8px;

      .li {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        row-gap: 8px;
        .p-box {
          max-width: 90%;
          padding: 10px 10px;
          font-size: 14px;
          min-height: 36px;
          color: #fff;
          border-radius: 36px;
          border: 1px solid rgba(24, 91, 197, 1);
          box-shadow: 0px 0px 6px 0px rgba(0, 213, 250, 1) inset;
          display: flex;
          align-items: center;
          justify-content: center;
          column-gap: 3px;
          box-sizing: border-box;
          cursor: pointer;

          img {
            width: 18px;
            height: 18px;
          }
          span {
            width: 100%;
            word-wrap: break-word;
          }
        }
        .active_item {
          background: rgba(25, 240, 255, 0.2);
        }
        .next {
          height: 24px;
        }
      }
    }
  }
}
</style>
