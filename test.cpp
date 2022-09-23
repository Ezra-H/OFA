#include <iostream>

public class Father{
public  static void main(String[] args){
Father father = new Father();
Child child = new Child();
try{
    father.test();
    child.test();}
catch (Exception e){
e.printStackTrace();
}

}
}