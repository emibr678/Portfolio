using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraScript : MonoBehaviour
{   
    float smooth = 0.01f; //0.01 - super smooth, 1 - super sharp 
    static public bool flyViewActive;
    // Start is called before the first frame update
    void Start()
    {
        flyViewActive = true;
    }

    void Update(){
            if (Input.GetKeyDown(KeyCode.Space)){
            if (flyViewActive){
                flyViewActive = false;
            } else {
                flyViewActive = true;
            }
        }
    }
    // Update is called once per frame
    void FixedUpdate()
    {
    
        // CAMERA OPTIONS
        if (flyViewActive == false){
            Vector3 moveCamTo = transform.position - transform.forward * 50.0f + Vector3.up * 10.0f;
            float bias = 0.96f; // To create a spring function
            Camera.main.transform.position = Camera.main.transform.position * bias + moveCamTo * (1.0f-bias);
            Camera.main.transform.LookAt(transform.position);
            Camera.main.transform.rotation = Quaternion.Lerp(Camera.main.transform.rotation, transform.rotation, smooth);
            Camera.main.transform.position = Vector3.Lerp(Camera.main.transform.position, transform.position + new Vector3(0, 2.0f, 0), smooth);
        }
    }
}
